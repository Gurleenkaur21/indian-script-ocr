"""preprocess.py — Image preprocessing for all 22 Indian scripts."""

import os, importlib, importlib.util
import cv2
import numpy as np

_SRC  = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("_lang", os.path.join(_SRC,"languages.py"))
_lang = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_lang)
is_rtl           = _lang.is_rtl
needs_shirorekha = _lang.needs_shirorekha


class Preprocessor:
    H, W = 32, 128

    def load(self, path):
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None: raise FileNotFoundError(f"Cannot open: {path}")
        return img

    def clahe(self, img):
        return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4,4)).apply(img)

    def remove_shadow(self, img):
        bg   = cv2.GaussianBlur(cv2.dilate(img,np.ones((5,5),np.uint8),iterations=5),(21,21),0)
        return np.clip(cv2.divide(img.astype(np.float32),bg.astype(np.float32),scale=255),0,255).astype(np.uint8)

    def denoise(self, img):
        return cv2.fastNlMeansDenoising(img,h=10,templateWindowSize=7,searchWindowSize=21)

    def deskew(self, img, rtl=False):
        coords = np.column_stack(np.where(img<128))
        if len(coords)<50: return img
        angle = cv2.minAreaRect(coords)[-1]
        if angle<-45: angle=-(90+angle)
        else: angle=-angle
        if abs(angle)<(8.0 if rtl else 0.5): return img
        h,w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w//2,h//2),angle,1.0)
        return cv2.warpAffine(img,M,(w,h),flags=cv2.INTER_CUBIC,borderMode=cv2.BORDER_REPLICATE)

    def binarize(self, img, printed=False):
        if printed:
            _,out = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        else:
            out = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,15,8)
        return out

    def remove_shirorekha(self, img):
        k = cv2.getStructuringElement(cv2.MORPH_RECT,(25,1))
        return cv2.subtract(img,cv2.morphologyEx(img,cv2.MORPH_OPEN,k))

    def mirror_rtl(self, img):
        return cv2.flip(img,1)

    def tight_crop(self, img, pad=4):
        coords = cv2.findNonZero(cv2.bitwise_not(img))
        if coords is None: return img
        x,y,w,h = cv2.boundingRect(coords)
        return img[max(0,y-pad):min(img.shape[0],y+h+pad),
                   max(0,x-pad):min(img.shape[1],x+w+pad)]

    def resize(self, img):
        h,w = img.shape[:2]
        if h==0 or w==0: return np.ones((self.H,self.W),np.uint8)*255
        nw = max(1,int(w*self.H/h))
        r  = cv2.resize(img,(nw,self.H),interpolation=cv2.INTER_AREA)
        if nw>=self.W: return cv2.resize(r,(self.W,self.H),interpolation=cv2.INTER_AREA)
        p = np.ones((self.H,self.W),dtype=np.uint8)*255; p[:,:nw]=r; return p

    def process(self, path, language="hindi", image_type="auto"):
        """Full pipeline → returns (32,128) uint8 numpy array."""
        rtl   = is_rtl(language)
        shiro = needs_shirorekha(language)
        img   = self.load(path)
        printed = float(np.std(img))<60.0 if image_type=="auto" else image_type=="printed"
        img = self.remove_shadow(img) if printed else self.clahe(img)
        img = self.denoise(img)
        img = self.deskew(img,rtl=rtl)
        img = self.binarize(img,printed=printed)
        if shiro: img = self.remove_shirorekha(img)
        if rtl:   img = self.mirror_rtl(img)
        img = self.tight_crop(img)
        return self.resize(img)

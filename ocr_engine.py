"""ocr_engine.py — Core OCR engine for all 22 Indian languages."""

import os, sys, time, importlib, importlib.util
from dataclasses import dataclass, field
from typing import Optional, List
import cv2, numpy as np

_SRC = os.path.dirname(os.path.abspath(__file__))

def _load(alias, filename):
    path = os.path.join(_SRC, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"\n\n  MISSING FILE: {path}"
            f"\n  Copy '{filename}' into folder: {_SRC}\n")
    spec = importlib.util.spec_from_file_location(alias, path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

# Check all required files exist first
_missing = [f for f in ["languages.py","preprocess.py","vocabulary.py","model.py"]
            if not os.path.exists(os.path.join(_SRC,f))]
if _missing:
    print(f"\nERROR — missing files in {_SRC}:")
    for f in _missing: print(f"  MISSING: {f}")
    print("\nCopy all 7 files from the ZIP into src/ folder.\n")
    sys.exit(1)

_lang = _load("_indocr_lang", "languages.py")
_prep = _load("_indocr_prep", "preprocess.py")

is_rtl         = _lang.is_rtl
easyocr_codes  = _lang.easyocr_codes
tesseract_code = _lang.tesseract_code
Preprocessor   = _prep.Preprocessor


@dataclass
class OCRResult:
    text:       str
    confidence: float
    language:   str
    backend:    str
    time_ms:    float
    boxes:      List = field(default_factory=list)
    error:      Optional[str] = None


class OCREngine:
    """
    Universal OCR for all 22 Indian languages.

    Usage:
        engine = OCREngine("hindi", "easyocr")
        result = engine.predict("image.jpg")
        print(result.text, result.confidence)
    """

    def __init__(self, language="hindi", backend="easyocr",
                 model_path=None, vocab_path=None, gpu=False):
        assert backend in ("easyocr","tesseract","custom"), \
            "backend must be: easyocr | tesseract | custom"
        self.language   = language.lower()
        self.backend    = backend
        self.gpu        = gpu
        self.rtl        = is_rtl(self.language)
        self.prep       = Preprocessor()
        self._reader    = None
        self.model_path = model_path
        self.vocab_path = vocab_path
        self._model     = None
        self._vocab     = None

    def _load_easyocr(self):
        if self._reader is None:
            import easyocr
            codes = easyocr_codes(self.language)
            print(f"[EasyOCR] Loading {self.language} {codes} ...")
            self._reader = easyocr.Reader(codes, gpu=self.gpu)
            print("[EasyOCR] Ready.")

    def _load_custom(self):
        if self._model is None:
            import torch
            _m = _load("_indocr_model", "model.py")
            _v = _load("_indocr_vocab", "vocabulary.py")
            dev = "cuda" if self.gpu and torch.cuda.is_available() else "cpu"
            self._device = dev
            self._vocab  = _v.Vocabulary().load(self.vocab_path)
            self._model  = _m.CRNN(num_classes=self._vocab.size).to(dev)
            ckpt = torch.load(self.model_path, map_location=dev)
            self._model.load_state_dict(ckpt["state"])
            self._model.eval()
            print(f"[Custom] Loaded — CER: {ckpt.get('cer',0)*100:.1f}%")

    def predict(self, image_path: str) -> OCRResult:
        """Extract text from one image file."""
        t0 = time.time()
        try:
            if   self.backend == "easyocr":   r = self._easyocr(image_path)
            elif self.backend == "tesseract":  r = self._tesseract(image_path)
            else:                              r = self._custom(image_path)
            r.time_ms = round((time.time()-t0)*1000,1); return r
        except Exception as e:
            return OCRResult("",0.0,self.language,self.backend,
                             round((time.time()-t0)*1000,1),error=str(e))

    def _easyocr(self, path):
        self._load_easyocr()
        raw   = self._reader.readtext(path)
        texts = [t for _,t,_ in raw]; confs=[c for _,_,c in raw]
        boxes = [{"text":t,"confidence":round(c*100,1),"box":b} for b,t,c in raw]
        text  = " ".join(texts)
        if self.rtl: text=text[::-1]
        avg = round(float(np.mean(confs))*100,1) if confs else 0.0
        return OCRResult(text,avg,self.language,"easyocr",0,boxes)

    def _tesseract(self, path):
        import pytesseract
        from PIL import Image
        img  = Image.open(path).convert("L")
        lang = tesseract_code(self.language)
        data = pytesseract.image_to_data(img,lang=lang,config="--oem 3 --psm 6",
               output_type=pytesseract.Output.DICT)
        words=[w for w,c in zip(data["text"],data["conf"])
               if w.strip() and str(c).lstrip("-").isdigit() and int(c)>0]
        confs=[int(c) for c in data["conf"]
               if str(c).lstrip("-").isdigit() and int(c)>0]
        text=" ".join(words)
        avg=round(float(np.mean(confs)),1) if confs else 0.0
        return OCRResult(text,avg,self.language,"tesseract",0)

    def _custom(self, path):
        import torch; self._load_custom()
        img = self.prep.process(path,self.language)
        t   = torch.FloatTensor(img.astype(np.float32)/255.0).unsqueeze(0).unsqueeze(0).to(self._device)
        with torch.no_grad():
            out  = self._model(t)
            conf = float(out.softmax(2).max(2).values.mean())*100
        _,best=out.max(2); seq=best.squeeze(1).cpu().tolist()
        col,prev=[],- 1
        for i in seq:
            if i!=prev: col.append(i)
            prev=i
        text=self._vocab.decode(col,rtl=self.rtl)
        return OCRResult(text,round(conf,1),self.language,"custom",0)

    def predict_array(self, img_array: np.ndarray) -> OCRResult:
        """Predict from an OpenCV / numpy image array."""
        tmp=os.path.join(_SRC,"_tmp_ocr.png"); cv2.imwrite(tmp,img_array)
        r=self.predict(tmp)
        try: os.remove(tmp)
        except: pass
        return r

    def predict_folder(self, folder, extensions=(".png",".jpg",".jpeg"), save_csv=None):
        """Run OCR on every image in a folder."""
        import glob
        paths=sorted(p for ext in extensions
                     for p in glob.glob(os.path.join(folder,f"**/*{ext}"),recursive=True))
        print(f"[Batch] {len(paths)} images in: {folder}")
        results=[]
        for i,p in enumerate(paths,1):
            r=self.predict(p); results.append(r)
            print(f"  [{i:4d}/{len(paths)}] {'OK ' if not r.error else 'ERR'}  "
                  f"{os.path.basename(p):40s}  {r.text[:35]!r}  ({r.confidence:.1f}%)")
        if save_csv:
            import pandas as pd
            d=os.path.dirname(os.path.abspath(save_csv))
            if d: os.makedirs(d,exist_ok=True)
            pd.DataFrame([{"filename":os.path.basename(p),"text":r.text,
                "confidence":r.confidence,"time_ms":r.time_ms,"error":r.error}
                for p,r in zip(paths,results)]).to_csv(save_csv,index=False,encoding="utf-8")
            print(f"[Batch] Saved → {save_csv}")
        return results


if __name__ == "__main__":
    import argparse
    ap=argparse.ArgumentParser(description="Indian OCR — extract text from image")
    ap.add_argument("image",      nargs="?", default=None, help="Path to image")
    ap.add_argument("--language", default="hindi", help="hindi/tamil/bengali etc")
    ap.add_argument("--backend",  default="easyocr",
                    choices=["easyocr","tesseract","custom"])
    ap.add_argument("--model", default="models/checkpoints/best_model.pth")
    ap.add_argument("--vocab", default="data/vocab/vocab.json")
    args = ap.parse_args()

    if not args.image:
        print("Usage examples:")
        print("  python src/ocr_engine.py photo.jpg")
        print("  python src/ocr_engine.py photo.jpg --language tamil")
        print("  python src/ocr_engine.py photo.jpg --backend tesseract")
        sys.exit(0)

    engine = OCREngine(args.language, args.backend, args.model, args.vocab)
    r = engine.predict(args.image)
    print(f"\nText       : {r.text}")
    print(f"Confidence : {r.confidence}%")
    print(f"Time       : {r.time_ms} ms")
    print(f"Language   : {r.language}")
    print(f"Backend    : {r.backend}")
    if r.error: print(f"Error      : {r.error}")


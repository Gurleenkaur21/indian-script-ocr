"""dataset.py — PyTorch Dataset and DataLoader for Indian OCR images."""

import os, sys, importlib, importlib.util, random
import cv2, numpy as np, pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

_SRC = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("_vocab", os.path.join(_SRC,"vocabulary.py"))
_vmod = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_vmod)
Vocabulary = _vmod.Vocabulary

class OCRDataset(Dataset):
    def __init__(self, split_dir, vocab, augment=False, H=32, W=128):
        self.vocab=vocab; self.augment=augment; self.H,self.W=H,W
        self.df=(pd.read_csv(os.path.join(split_dir,"labels.csv"),encoding="utf-8")
                   .dropna(subset=["filename","text"]).reset_index(drop=True))
        self.img_dir=os.path.join(split_dir,"images")
        print(f"[Dataset] {len(self.df)} samples from {split_dir}")

    def __len__(self): return len(self.df)

    def __getitem__(self,idx):
        row=self.df.iloc[idx]
        img=cv2.imread(os.path.join(self.img_dir,str(row["filename"])),cv2.IMREAD_GRAYSCALE)
        if img is None: img=np.ones((self.H,self.W),dtype=np.uint8)*255
        if img.shape!=(self.H,self.W): img=cv2.resize(img,(self.W,self.H))
        if self.augment: img=_aug(img)
        img_t=torch.FloatTensor(img.astype(np.float32)/255.0).unsqueeze(0)
        label=self.vocab.encode(str(row["text"]))
        return img_t, torch.LongTensor(label), len(label)

def _aug(img):
    if random.random()<0.4:
        img=np.clip(img.astype(np.int16)+random.randint(-35,35),0,255).astype(np.uint8)
    if random.random()<0.35:
        img=np.clip(img.astype(np.int16)+np.random.normal(0,10,img.shape).astype(np.int16),0,255).astype(np.uint8)
    if random.random()<0.4:
        a=random.uniform(-6,6); h,w=img.shape
        M=cv2.getRotationMatrix2D((w//2,h//2),a,1.0)
        img=cv2.warpAffine(img,M,(w,h),borderMode=cv2.BORDER_REPLICATE)
    return img

def collate_fn(batch):
    images,labels,lengths=zip(*batch)
    images=torch.stack(images,0); max_len=max(lengths)
    padded=torch.zeros(len(labels),max_len,dtype=torch.long)
    for i,(lbl,ln) in enumerate(zip(labels,lengths)): padded[i,:ln]=lbl
    return images,padded,torch.LongTensor(lengths)

def make_loaders(data_dir,vocab,batch_size=32,num_workers=4):
    kw=dict(collate_fn=collate_fn,num_workers=num_workers,pin_memory=True)
    tr=OCRDataset(f"{data_dir}/train",vocab,augment=True)
    va=OCRDataset(f"{data_dir}/val",  vocab,augment=False)
    te=OCRDataset(f"{data_dir}/test", vocab,augment=False)
    return (DataLoader(tr,batch_size,shuffle=True, **kw),
            DataLoader(va,batch_size,shuffle=False,**kw),
            DataLoader(te,batch_size,shuffle=False,**kw))
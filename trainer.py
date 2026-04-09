"""train/trainer.py — Full CRNN training loop.  Run: python train/trainer.py"""

import os, sys, json, importlib, importlib.util

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC  = os.path.join(_ROOT,"src")

def _load(alias,fname):
    spec=importlib.util.spec_from_file_location(alias,os.path.join(_SRC,fname))
    mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod

_ds  = _load("_tr_dataset",   "dataset.py")
_md  = _load("_tr_model",     "model.py")
_voc = _load("_tr_vocabulary","vocabulary.py")

make_loaders = _ds.make_loaders
build_model  = _md.build_model
Vocabulary   = _voc.Vocabulary

import torch, torch.nn as nn, torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
try:
    import editdistance
except ImportError:
    import Levenshtein as editdistance

try: from tqdm import tqdm; TQDM=True
except: TQDM=False

# ── CONFIG
CFG = {
    "data_dir":   "data/processed",
    "vocab_path": "data/vocab/vocab.json",
    "save_dir":   "models/checkpoints",
    "batch_size": 32,
    "epochs":     100,
    "early_stop": 15,
    "lr":         1e-3,
    "num_workers":0,       # use 0 on Windows to avoid errors
    "gpu":        True,
}

def _iter(loader, desc):
    return tqdm(loader,desc=desc,leave=False) if TQDM else loader

def greedy_decode(output, vocab):
    _,best=output.max(2); best=best.T.cpu().numpy(); preds=[]
    for seq in best:
        col,prev=[],- 1
        for idx in seq:
            if idx!=prev: col.append(int(idx))
            prev=idx
        preds.append(vocab.decode(col))
    return preds

def cer(preds,targets):
    t=c=0
    for p,g in zip(preds,targets):
        if not g: continue
        t+=editdistance.eval(p,g)/len(g); c+=1
    return t/c if c else 1.0

def train():
    device="cuda" if CFG["gpu"] and torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")

    vocab=Vocabulary().load(CFG["vocab_path"])
    tl,vl,tel=make_loaders(CFG["data_dir"],vocab,CFG["batch_size"],CFG["num_workers"])
    model=build_model(vocab.size,device)
    ctc=nn.CTCLoss(blank=0,zero_infinity=True)
    opt=optim.Adam(model.parameters(),lr=CFG["lr"],weight_decay=1e-4)
    sch=ReduceLROnPlateau(opt,"min",patience=5,factor=0.5,min_lr=1e-6,verbose=True)
    os.makedirs(CFG["save_dir"],exist_ok=True)
    best_cer=float("inf"); no_imp=0

    for epoch in range(1,CFG["epochs"]+1):
        # Train
        model.train(); tl_sum=0
        for imgs,labels,lengths in _iter(tl,f"Ep{epoch:03d} Train"):
            imgs=imgs.to(device); labels=labels.to(device)
            out=model(imgs); B,T=imgs.size(0),out.size(0)
            il=torch.full((B,),T,dtype=torch.long)
            loss=ctc(out.log_softmax(2),labels.view(-1),il,lengths)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),5)
            opt.step(); tl_sum+=loss.item()
        tr_loss=tl_sum/len(tl)

        # Val
        model.eval(); vl_sum=0; all_p,all_g=[],[]
        with torch.no_grad():
            for imgs,labels,lengths in _iter(vl,f"Ep{epoch:03d} Val  "):
                imgs=imgs.to(device); labels=labels.to(device)
                out=model(imgs); B,T=imgs.size(0),out.size(0)
                il=torch.full((B,),T,dtype=torch.long)
                vl_sum+=ctc(out.log_softmax(2),labels.view(-1),il,lengths).item()
                all_p+=greedy_decode(out,vocab)
                for i,ln in enumerate(lengths):
                    all_g.append(vocab.decode(labels[i,:ln].tolist()))
        val_loss=vl_sum/len(vl); val_cer=cer(all_p,all_g)
        sch.step(val_cer)

        flag="★ BEST" if val_cer<best_cer else ""
        print(f"Ep{epoch:03d} | train={tr_loss:.4f}  val={val_loss:.4f}  "
              f"CER={val_cer*100:.2f}%  {flag}")
        for p,g in zip(all_p[:2],all_g[:2]):
            print(f"  [{'OK' if p==g else '??'}] GT:{g!r:20s} → Pred:{p!r}")

        if val_cer<best_cer:
            best_cer=val_cer; no_imp=0
            torch.save({"epoch":epoch,"state":model.state_dict(),"cer":val_cer},
                       os.path.join(CFG["save_dir"],"best_model.pth"))
        else:
            no_imp+=1
            if no_imp>=CFG["early_stop"]:
                print(f"Early stopping at epoch {epoch}"); break

    print(f"\nBest Val CER: {best_cer*100:.2f}%")
    print(f"Model saved: {CFG['save_dir']}/best_model.pth")

if __name__=="__main__": train()
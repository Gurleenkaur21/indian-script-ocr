"""vocabulary.py — Character to integer mapping for Indian scripts."""

import json, os
from collections import Counter

class Vocabulary:
    def __init__(self):
        self.char2idx = {"<BLANK>":0,"<UNK>":1}
        self.idx2char = {0:"<BLANK>",1:"<UNK>"}
        self.blank, self.unk = 0, 1

    def build(self, csv_paths, min_freq=3):
        import pandas as pd
        all_text = ""
        for p in csv_paths:
            if os.path.exists(p):
                all_text += " ".join(pd.read_csv(p,encoding="utf-8")["text"].astype(str))
        valid = sorted([c for c,n in Counter(all_text).items() if n>=min_freq and c.strip()])
        for i,c in enumerate(valid,start=2):
            self.char2idx[c]=i; self.idx2char[i]=c
        print(f"Vocabulary: {len(valid)} chars | {self.size} total classes")
        return self

    @property
    def size(self): return len(self.char2idx)

    def encode(self, text):
        return [self.char2idx.get(c,self.unk) for c in str(text)]

    def decode(self, indices, rtl=False):
        result,prev=[],- 1
        for idx in indices:
            if idx!=prev and idx not in (self.blank,self.unk):
                c=self.idx2char.get(idx,"")
                if c: result.append(c)
            prev=idx
        text="".join(result)
        return text[::-1] if rtl else text

    def save(self, path="data/vocab/vocab.json"):
        os.makedirs(os.path.dirname(os.path.abspath(path)),exist_ok=True)
        with open(path,"w",encoding="utf-8") as f:
            json.dump({"char2idx":self.char2idx},f,ensure_ascii=False,indent=2)
        print(f"Vocabulary saved → {path}"); return self

    def load(self, path="data/vocab/vocab.json"):
        with open(path,encoding="utf-8") as f:
            self.char2idx=json.load(f)["char2idx"]
        self.idx2char={int(v):k for k,v in self.char2idx.items()}
        print(f"Vocabulary loaded: {self.size} classes"); return self
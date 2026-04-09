"""model.py — CRNN model for Indian OCR training."""

import torch, torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self,in_ch,out_ch,pool=None):
        super().__init__()
        layers=[nn.Conv2d(in_ch,out_ch,3,padding=1),nn.BatchNorm2d(out_ch),nn.ReLU(True)]
        if pool: layers.append(nn.MaxPool2d(pool))
        self.net=nn.Sequential(*layers)
    def forward(self,x): return self.net(x)

class BiLSTM(nn.Module):
    def __init__(self,in_s,hidden,out_s):
        super().__init__()
        self.rnn=nn.LSTM(in_s,hidden,bidirectional=True,batch_first=False)
        self.fc=nn.Linear(hidden*2,out_s)
    def forward(self,x):
        out,_=self.rnn(x); return self.fc(out)

class CRNN(nn.Module):
    """Input: (B,1,32,128) → Output: (T,B,num_classes)"""
    def __init__(self,num_classes,rnn_hidden=256):
        super().__init__()
        self.cnn=nn.Sequential(
            ConvBlock(1,  64, pool=(2,2)),
            ConvBlock(64, 128,pool=(2,2)),
            ConvBlock(128,256),
            ConvBlock(256,256,pool=(2,1)),
            ConvBlock(256,512),
            ConvBlock(512,512,pool=(2,1)),
            nn.Conv2d(512,512,kernel_size=2),
            nn.BatchNorm2d(512),nn.ReLU(True),
        )
        self.rnn=nn.Sequential(
            BiLSTM(512,rnn_hidden,rnn_hidden),
            BiLSTM(rnn_hidden,rnn_hidden,num_classes),
        )
        self.drop=nn.Dropout(0.25)
    def forward(self,x):
        f=self.cnn(x).squeeze(2).permute(2,0,1)
        return self.rnn(self.drop(f))

def build_model(num_classes,device="cpu"):
    m=CRNN(num_classes).to(device)
    print(f"Model: {sum(p.numel() for p in m.parameters()):,} params | device={device}")
    return m
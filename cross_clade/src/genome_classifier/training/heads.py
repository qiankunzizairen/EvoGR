import torch
from torch import nn

class BaselineHead(nn.Module):
    def __init__(self, num_classes=4, dropout=0.1):
        super().__init__(); self.net = nn.Sequential(nn.LayerNorm(4096), nn.Linear(4096,1024), nn.GELU(), nn.Dropout(dropout), nn.Linear(1024,256), nn.GELU(), nn.Dropout(dropout), nn.Linear(256,num_classes))
    def forward(self, x): return self.net(x.mean(dim=1))

class TransformerHead(nn.Module):
    def __init__(self, num_classes=4, dropout=0.1):
        super().__init__(); self.proj=nn.Sequential(nn.Linear(4096,256),nn.LayerNorm(256)); layer=nn.TransformerEncoderLayer(256,8,dim_feedforward=1024,dropout=dropout,activation="gelu",batch_first=True); self.encoder=nn.TransformerEncoder(layer,2); self.mlp=nn.Sequential(nn.Linear(256,128),nn.GELU(),nn.Dropout(dropout),nn.Linear(128,num_classes))
    def forward(self,x): return self.mlp(self.encoder(self.proj(x)).mean(dim=1))

def build_head(name, num_classes=4, dropout=0.1):
    if name == "baseline": return BaselineHead(num_classes, dropout)
    if name == "transformer": return TransformerHead(num_classes, dropout)
    raise ValueError(f"Unknown head: {name}")

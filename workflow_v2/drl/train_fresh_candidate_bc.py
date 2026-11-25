#!/usr/bin/env python3
"""Fit fresh candidate-index BC from WiLab causal-joint demonstrations."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parent
DATA=ROOT.parent/'outputs'/'p3_bc_dataset_v3'
OUT=ROOT/'outputs'; OUT.mkdir(exist_ok=True)

class Scorer(nn.Module):
    def __init__(self):
        super().__init__(); self.net=nn.Sequential(nn.Linear(15,96),nn.ReLU(),nn.Linear(96,1))
    def forward(self,x): return self.net(x).squeeze(-1)

def load():
    rows=[]
    for p in DATA.glob('**/fresh_demos.json'):
        demos=json.loads(p.read_text())
        for d in demos:
            x=np.asarray(d['candidate_features'],np.float32)
            y=int(d['action_index'])-1
            if x.ndim==2 and 0<=y<len(x): rows.append((x,y))
    if not rows: raise RuntimeError('no complete fresh candidate demonstrations')
    return rows

def main():
    torch.manual_seed(17); np.random.seed(17); data=load(); model=Scorer(); opt=torch.optim.Adam(model.parameters(),lr=1e-3)
    for epoch in range(80):
        np.random.shuffle(data); losses=[]; correct=0
        for x,y in data:
            q=model(torch.tensor(x)); loss=F.cross_entropy(q[None,:],torch.tensor([y])); opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item()); correct+=int(q.argmax().item()==y)
        if epoch in (0,19,39,79): print(f'epoch={epoch+1} loss={np.mean(losses):.4f} acc={correct/len(data):.3f}',flush=True)
    torch.save(model.state_dict(),OUT/'fresh_candidate_bc.pt')
    payload={'input_dim':15,'hidden':96,'state_dict':{k:v.detach().tolist() for k,v in model.state_dict().items()},'examples':len(data)}
    (OUT/'fresh_candidate_bc.json').write_text(json.dumps(payload),encoding='utf-8')
    print(json.dumps({'examples':len(data),'train_accuracy':correct/len(data)}))
if __name__=='__main__': main()

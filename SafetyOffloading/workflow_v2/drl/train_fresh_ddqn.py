#!/usr/bin/env python3
"""Double-DQN fine-tuning for variable fresh WiLab action candidates."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parent
TRANS=ROOT/'outputs'/'fresh_ddqn_transitions.npz'; BC=ROOT/'outputs'/'fresh_candidate_bc.json'; OUT=ROOT/'outputs'/'fresh_ddqn.json'
class Q(nn.Module):
 def __init__(self): super().__init__(); self.net=nn.Sequential(nn.Linear(15,96),nn.ReLU(),nn.Linear(96,1))
 def forward(self,x): return self.net(x).squeeze(-1)
def load_bc(m):
 d=json.loads(BC.read_text())['state_dict']; m.load_state_dict({k:torch.tensor(v) for k,v in d.items()})
def main():
 torch.manual_seed(23); data=np.load(TRANS,allow_pickle=True)['transitions']; q=Q(); load_bc(q); target=Q(); target.load_state_dict(q.state_dict()); opt=torch.optim.Adam(q.parameters(),lr=2e-4); gamma=.99
 for ep in range(300):
  order=np.random.permutation(len(data)); losses=[]
  for j in order:
   cur,a,nxt,r,done=data[j]; cur=torch.tensor(cur); nxt=torch.tensor(nxt)
   qa=q(cur)[int(a)]
   with torch.no_grad():
    if done or len(nxt)==0: y=torch.tensor(float(r))
    else:
     online=q(nxt); pick=int(online.argmax()); y=torch.tensor(float(r))+gamma*target(nxt)[pick]
   loss=F.smooth_l1_loss(qa,y); opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(q.parameters(),5); opt.step(); losses.append(loss.item())
  if (ep+1)%25==0: target.load_state_dict(q.state_dict()); print(f'epoch={ep+1} loss={np.mean(losses):.5f}',flush=True)
 sd={k:v.detach().tolist() for k,v in q.state_dict().items()}; OUT.write_text(json.dumps({'input_dim':15,'hidden':96,'state_dict':sd,'source':'fresh_wilab_ddqn'})); print(OUT)
if __name__=='__main__': main()

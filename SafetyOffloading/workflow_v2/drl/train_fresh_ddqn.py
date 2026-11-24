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
DEMO=ROOT.parent/'outputs'/'p3_bc_dataset_v3'
class Q(nn.Module):
 def __init__(self): super().__init__(); self.net=nn.Sequential(nn.Linear(15,96),nn.ReLU(),nn.Linear(96,1))
 def forward(self,x): return self.net(x).squeeze(-1)
def load_bc(m):
 d=json.loads(BC.read_text())['state_dict']; m.load_state_dict({k:torch.tensor(v) for k,v in d.items()})
def load_demos():
 out=[]
 for p in DEMO.glob('**/fresh_demos.json'):
  for d in json.loads(p.read_text()):
   x=np.asarray(d['candidate_features'],np.float32)
   if x.ndim==1: x=x[None,:]
   y=int(d['action_index'])-1
   if 0<=y<len(x): out.append((x,y))
 return out
def main():
 torch.manual_seed(23); data=np.load(TRANS,allow_pickle=True)['transitions']; demos=load_demos(); q=Q(); load_bc(q); target=Q(); target.load_state_dict(q.state_dict()); opt=torch.optim.Adam(q.parameters(),lr=1e-4); gamma=.99
 for ep in range(160):
  order=np.random.permutation(len(data)); losses=[]
  for j in order:
   cur,a,nxt,r,done=data[j]; cur=torch.tensor(cur); nxt=torch.tensor(nxt)
   qa=q(cur)[int(a)]
   with torch.no_grad():
    if done or len(nxt)==0: y=torch.tensor(float(r))
    else:
     online=q(nxt); pick=int(online.argmax()); y=torch.tensor(float(r))+gamma*target(nxt)[pick]
   loss=F.smooth_l1_loss(qa,y); opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(q.parameters(),5); opt.step(); losses.append(loss.item())
  # Conservative BC anchor prevents unsupported candidate values from erasing
  # the physically validated joint-placement behavior.
  for x,y in [demos[k] for k in np.random.choice(len(demos),min(128,len(demos)),replace=False)]:
   logits=q(torch.tensor(x))[None,:]; loss=0.5*F.cross_entropy(logits,torch.tensor([y])); opt.zero_grad(); loss.backward(); opt.step()
  if (ep+1)%20==0: target.load_state_dict(q.state_dict()); print(f'epoch={ep+1} td_loss={np.mean(losses):.5f}',flush=True)
 sd={k:v.detach().tolist() for k,v in q.state_dict().items()}; OUT.write_text(json.dumps({'input_dim':15,'hidden':96,'state_dict':sd,'source':'fresh_wilab_ddqn'})); print(OUT)
if __name__=='__main__': main()

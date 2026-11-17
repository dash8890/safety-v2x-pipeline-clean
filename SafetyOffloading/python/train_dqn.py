#!/usr/bin/env python3
"""Offline DQN for task admission and ranked helper selection."""
from __future__ import annotations
import argparse, csv, json, math, random
from pathlib import Path

FEATURES=["cbr","density","candidate_helpers","helper_distance_m","origin_queue_length",
          "helper_queue_length","input_bytes","output_bytes","cycles","deadline_ms"]
ACTIONS=["reject","local","helper_1","helper_2","helper_3","helper_4"]

def num(r,k):
    try:
        v=float(r.get(k,0)); return v if math.isfinite(v) else 0.0
    except ValueError: return 1.0 if str(r.get(k,"")).lower() in {"true","1"} else 0.0
def relu(x): return x if x>0 else 0.0
def forward(x,w1,b1,w2,b2):
    h=[relu(sum(w*v for w,v in zip(row,x))+b) for row,b in zip(w1,b1)]
    q=[sum(w*v for w,v in zip(row,h))+b for row,b in zip(w2,b2)]
    return h,q
def valid_actions(row): return list(range(min(len(ACTIONS),2+int(num(row,"candidate_helpers")))))

def main():
    p=argparse.ArgumentParser(); p.add_argument("dataset",type=Path); p.add_argument("--output",type=Path,required=True)
    p.add_argument("--epochs",type=int,default=400); p.add_argument("--hidden",type=int,default=32)
    p.add_argument("--seed",type=int,default=17); a=p.parse_args(); random.seed(a.seed)
    rows=list(csv.DictReader(a.dataset.open(encoding="utf-8-sig",newline="")))
    missing=set(FEATURES+["seed","scenario_id","vehicle_id","time_ms","action_index","completed_on_time"])-set(rows[0])
    if missing: raise ValueError(f"missing columns: {sorted(missing)}")
    test_seed=max(int(num(r,"seed")) for r in rows); train=[r for r in rows if int(num(r,"seed"))!=test_seed]; test=[r for r in rows if int(num(r,"seed"))==test_seed]
    means=[sum(num(r,f) for r in train)/len(train) for f in FEATURES]
    scales=[max(math.sqrt(sum((num(r,f)-m)**2 for r in train)/max(1,len(train)-1)),1e-12) for f,m in zip(FEATURES,means)]
    def state(r): return [(num(r,f)-m)/s for f,m,s in zip(FEATURES,means,scales)]
    def transitions(data):
        ordered=sorted(data,key=lambda r:(int(num(r,"scenario_id")),int(num(r,"vehicle_id")),num(r,"time_ms")))
        out=[]
        for i,r in enumerate(ordered):
            nxt=None
            if i+1<len(ordered) and ordered[i+1]["scenario_id"]==r["scenario_id"] and ordered[i+1]["vehicle_id"]==r["vehicle_id"]: nxt=ordered[i+1]
            out.append((r,nxt))
        return out
    tr=transitions(train); n=len(FEATURES); h=a.hidden; na=len(ACTIONS)
    w1=[[random.uniform(-0.1,0.1) for _ in range(n)] for _ in range(h)]; b1=[0.0]*h
    w2=[[random.uniform(-0.1,0.1) for _ in range(h)] for _ in range(na)]; b2=[0.0]*na
    tw1=[r[:] for r in w1]; tb1=b1[:]; tw2=[r[:] for r in w2]; tb2=b2[:]
    gamma=.95; lr=.002
    for epoch in range(a.epochs):
        random.shuffle(tr)
        for r,nxt in tr:
            x=state(r); hidden,q=forward(x,w1,b1,w2,b2); action=int(num(r,"action_index")); reward=num(r,"completed_on_time")
            target=reward
            if nxt is not None:
                _,nq=forward(state(nxt),tw1,tb1,tw2,tb2); target+=gamma*max(nq[j] for j in valid_actions(nxt))
            error=max(-5.0,min(5.0,q[action]-target)); old=w2[action][:]
            for j in range(h): w2[action][j]-=lr*error*hidden[j]
            b2[action]-=lr*error
            for j in range(h):
                if hidden[j]>0:
                    grad=error*old[j]
                    for k in range(n): w1[j][k]-=lr*grad*x[k]
                    b1[j]-=lr*grad
        if (epoch+1)%20==0: tw1=[r[:] for r in w1];tb1=b1[:];tw2=[r[:] for r in w2];tb2=b2[:]
    def evaluate(data):
        correct=0; chosen=[0]*na
        for r in data:
            _,q=forward(state(r),w1,b1,w2,b2); action=max(valid_actions(r),key=lambda j:q[j]); chosen[action]+=1
            correct+=int(action==int(num(r,"action_index")))
        return {"rows":len(data),"behavior_action_match":correct/max(1,len(data)),"chosen_actions":dict(zip(ACTIONS,chosen))}
    artifact={"model":"offline_dqn","features":FEATURES,"actions":ACTIONS,"means":means,"scales":scales,
              "hidden_weights":w1,"hidden_bias":b1,"output_weights":w2,"output_bias":b2,
              "training":{"epochs":a.epochs,"discount":gamma,"learning_rate":lr,"train_rows":len(train),"held_out_seed":test_seed},
              "evaluation":evaluate(test),"risk_shield_required":True}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(artifact,indent=2),encoding="utf-8")
    print(json.dumps(artifact["evaluation"],indent=2))
if __name__=="__main__": main()

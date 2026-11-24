#!/usr/bin/env python3
"""Build variable-candidate TD trajectories from physical WiLab traces."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent.parent
SOURCES=[ROOT/'outputs'/'p4_bc_traces', ROOT/'outputs'/'p4_explore_traces']
OUT=Path(__file__).resolve().parent/'outputs'/'fresh_ddqn_transitions.npz'

def main():
    traces=[]
    for source in SOURCES:
      for p in source.glob('**/fresh_trace.json'):
        d=json.loads(p.read_text()); dec=d['decisions']; terminal=float(d['terminal_utility'])
        # STOP_PLAN is a simulator/controller no-op, not an executable
        # scheduling decision. Older logs contain one at nearly every tick;
        # exclude them before forming the physical decision sequence.
        dec=[x for x in dec if x.get('kind') != 'STOP_PLAN']
        for i,x in enumerate(dec):
            cur=np.asarray(x['candidate_features'],np.float32)
            if cur.ndim==1: cur=cur[None,:]
            action=int(x['action_index'])-1
            nxt=np.asarray(dec[i+1]['candidate_features'],np.float32) if i+1<len(dec) else np.zeros((0,cur.shape[1]),np.float32)
            if nxt.ndim==1: nxt=nxt[None,:]
            # Delayed physical utility: incremental realized utility, with the
            # terminal completion value credited only at episode end.
            u0=float(x['utility_before']); u1=float(dec[i+1]['utility_before']) if i+1<len(dec) else terminal
            traces.append((cur,action,nxt,u1-u0,float(i+1==len(dec))))
    if not traces: raise RuntimeError('no fresh WiLab trace found')
    OUT.parent.mkdir(exist_ok=True)
    np.savez_compressed(OUT,transitions=np.asarray(traces,dtype=object))
    print(json.dumps({'transitions':len(traces),'output':str(OUT)}))
if __name__=='__main__': main()

#!/usr/bin/env python3
"""Build the sole public reward ledger for the frozen 80-rollout protocol."""
from __future__ import annotations
import json
import argparse
from pathlib import Path

ROOT=Path(__file__).resolve().parent.parent
parser=argparse.ArgumentParser()
parser.add_argument('--source',type=Path,default=ROOT/'outputs'/'p4_online80'/'episodes')
parser.add_argument('--output',type=Path,default=ROOT/'outputs'/'p4_online80'/'episode_rewards.json')
args=parser.parse_args()
EP=args.source; OUT=args.output
rows=[]
for path in sorted(EP.glob('rollout*/workflow_v2/summary.json')):
    s=json.loads(path.read_text()); number=int(path.parents[1].name.replace('rollout',''))
    rows.append({'rollout':number,'physical_utility':float(s['total_value']),
                 'n_success':int(s['n_success']),'n_expired':int(s['n_expired']),
                 'seed':int(s['seed'])})
rows.sort(key=lambda x:x['episode'])
OUT.parent.mkdir(parents=True,exist_ok=True)
OUT.write_text(json.dumps({'metric':'physical_utility_total_value','window':10,
                           'rollouts':rows},indent=2))
print(json.dumps({'rollouts':len(rows),'output':str(OUT)}))

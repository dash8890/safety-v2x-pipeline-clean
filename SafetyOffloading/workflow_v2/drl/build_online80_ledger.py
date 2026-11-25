#!/usr/bin/env python3
"""Build the sole public reward ledger for the frozen 80-rollout protocol."""
from __future__ import annotations
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent.parent
EP=ROOT/'outputs'/'p4_online80'/'episodes'
OUT=ROOT/'outputs'/'p4_online80'/'episode_rewards.json'
rows=[]
for path in sorted(EP.glob('episode*/workflow_v2/summary.json')):
    s=json.loads(path.read_text()); number=int(path.parents[1].name.replace('episode',''))
    rows.append({'episode':number,'physical_utility':float(s['total_value']),
                 'n_success':int(s['n_success']),'n_expired':int(s['n_expired']),
                 'seed':int(s['seed'])})
rows.sort(key=lambda x:x['episode'])
OUT.parent.mkdir(parents=True,exist_ok=True)
OUT.write_text(json.dumps({'metric':'physical_utility_total_value','window':10,
                           'episodes':rows},indent=2))
print(json.dumps({'episodes':len(rows),'output':str(OUT)}))

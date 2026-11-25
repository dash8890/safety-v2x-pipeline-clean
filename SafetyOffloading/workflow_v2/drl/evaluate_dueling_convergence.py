#!/usr/bin/env python3
"""Evaluate only training-derived Dueling-DDQN convergence evidence."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np

def mean_windows(values, size):
    return np.asarray([np.mean(values[i:i + size]) for i in range(0, len(values) - size + 1, size)])

def main():
    p=argparse.ArgumentParser(); p.add_argument('--run-root',type=Path,required=True); p.add_argument('--output',type=Path,required=True); a=p.parse_args()
    probes=[]; updates=[]
    for report in sorted((a.run_root/'training').glob('update*/optimizer_updates.json')):
        d=json.loads(report.read_text())
        if d.get('architecture') != 'dueling_ddqn': raise RuntimeError(f'non-dueling report: {report}')
        updates.extend(d['history'])
    # Probe means are the fixed-set, unshaped physical utilities.
    for manifest in sorted((a.run_root/'probes').glob('update*/update*/probe_manifest.json')):
        returns=[]
        for summary in manifest.parent.glob('seed*/workflow_v2/summary.json'):
            returns.append(float(json.loads(summary.read_text())['total_value']))
        if len(returns)==5: probes.append({'checkpoint':str(manifest.parent.relative_to(a.run_root)), 'mean':float(np.mean(returns)), 'std':float(np.std(returns)), 'returns':returns})
    utility=np.asarray([x['mean'] for x in probes]); n=len(utility); w=max(1,int(np.ceil(.10*n)))
    windows=mean_windows(utility,w)
    first=float(np.mean(utility[:w])); latest=float(np.mean(utility[-w:])); best=float(windows.max()) if len(windows) else float('-inf')
    gains=np.diff(windows)
    progressive = len(windows)>=4 and latest >= 1.20*first and np.count_nonzero(gains>0)>=3
    refine = len(windows)>=4 and all(x>=.95*best for x in windows[-3:]) and abs(gains[-1])<=abs(gains[-2])<=abs(gains[-3])
    final=windows[-max(2,int(np.ceil(.20*len(windows)))):]
    stable_perf = len(final)>=2 and (final.max()-final.min()) <= .05*float(final.mean()) and final[-1]>=final[0]*.95
    q=np.asarray([x['q_p50'] for x in updates]); td=np.asarray([x['mean_td_loss'] for x in updates]); agree=np.asarray([x['fixed_state_action_agreement'] for x in updates])
    tail=max(1,int(np.ceil(.20*len(updates))))
    stable_values = np.all(np.isfinite(q)) and np.all(np.isfinite(td)) and np.mean(td[-tail:]) <= 1.05*np.mean(td[-2*tail:-tail]) and np.max(np.abs(q[-tail:])) <= 1.10*np.max(np.abs(q[-2*tail:-tail]))
    stable_policy = bool(np.min(agree[-tail:]) >= .90)
    retention = latest >= .95*best
    extension = [x['mean'] for x in probes if 'extension' in x['checkpoint']]
    premature = bool(extension and np.mean(extension[-w:]) > 1.05*best)
    out={'run_root':str(a.run_root),'n_fixed_evaluations':n,'window_size':w,'first_window_mean':first,'latest_window_mean':latest,'best_window_mean':best,'window_means':windows.tolist(), 'stage1_progressive_learning':bool(progressive),'stage2_policy_refinement':bool(refine),'stage3_performance_stability':bool(stable_perf),'stage3_policy_stability':bool(stable_policy),'stage3_value_learning_stability':bool(stable_values),'stage3_performance_retention':bool(retention),'premature_stabilization':premature,'converged':bool(progressive and refine and stable_perf and stable_policy and stable_values and retention and not premature)}
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(out,indent=2)); print(json.dumps(out))
if __name__=='__main__': main()

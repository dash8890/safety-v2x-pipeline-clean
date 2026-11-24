#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

BASE = Path(__file__).resolve().parents[1] / "outputs" / "wilab_ddqn_convergence"
CASES = {
    4: 5104,
    10: 5110,
    13: 5113,
}
REF = {9: 5109, 11: 5111, 12: 5112}


def load(ep: int, seed: int):
    p = BASE / f"train_ep{ep:03d}_seed{seed}" / "workflow_v2"
    s = json.loads((p / "summary.json").read_text(encoding="utf-8"))
    w = list(csv.DictReader((p / "workflows.csv").open(encoding="utf-8")))
    d = list(csv.DictReader((p / "decisions.csv").open(encoding="utf-8")))
    return s, w, d


def analyze(ep: int, seed: int) -> None:
    s, w, d = load(ep, seed)
    succ = [x for x in w if x["status"] == "SUCCEEDED"]
    high = [x for x in w if float(x["value"]) >= 12]
    high_succ = [x for x in high if x["status"] == "SUCCEEDED"]
    high_exp = [x for x in high if x["status"] == "EXPIRED"]
    acts = Counter(x["action"] for x in d)
    drops = [x for x in d if x["action"] == "DROP"]
    drop_high = []
    high_ids = {x["workflow_id"] for x in high}
    for x in drops:
        if x["workflow_id"] in high_ids:
            drop_high.append((x["workflow_id"], x.get("detail", "")))
    early_high = [x for x in high if float(x["release_s"]) < 0.16]
    # first decision on each high
    first = {}
    for x in d:
        wid = x["workflow_id"]
        if wid in high_ids and wid not in first and x["action"] != "RELEASE_EDGE":
            first[wid] = (x.get("t_s", ""), x["action"], x.get("detail", ""))
    print(f"==== EP{ep} seed={seed} V={s['total_value']} succ={s['n_success']} exp={s['n_expired']} "
          f"frags={s.get('radio_data_fragments')} remote={s.get('n_remote_stages_done')} "
          f"RELEASE_EDGE={acts.get('RELEASE_EDGE', 0)}")
    print(f"  high total={len(high)} succ={len(high_succ)} exp={len(high_exp)} early_phase_high(<0.16s)={len(early_high)}")
    print(f"  SUCCEEDED:", [(x["workflow_id"], x["value"]) for x in succ])
    print(f"  early highs:", [(x["workflow_id"], x["value"], x["release_s"], x["status"]) for x in early_high])
    print(f"  first action on highs:", sorted(first.items(), key=lambda kv: float(kv[1][0] or 0)))
    print(f"  DROP highs:", drop_high)
    print(f"  actions:", dict(acts))


def main() -> None:
    for ep, seed in sorted({**CASES, **REF}.items()):
        path = BASE / f"train_ep{ep:03d}_seed{seed}" / "workflow_v2" / "summary.json"
        if not path.is_file():
            print(f"EP{ep} missing")
            continue
        analyze(ep, seed)
        print()


if __name__ == "__main__":
    main()

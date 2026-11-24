#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

BASE = Path(__file__).resolve().parents[1] / "outputs" / "wilab_ddqn_convergence"


def load(ep: int, seed: int):
    p = BASE / f"train_ep{ep:03d}_seed{seed}" / "workflow_v2"
    s = json.loads((p / "summary.json").read_text(encoding="utf-8"))
    w = list(csv.DictReader((p / "workflows.csv").open(encoding="utf-8")))
    d = list(csv.DictReader((p / "decisions.csv").open(encoding="utf-8")))
    return s, w, d


def main() -> None:
    for ep, seed in [(9, 5109), (10, 5110), (11, 5111)]:
        try:
            s, w, d = load(ep, seed)
        except Exception as e:
            print(f"EP{ep} MISSING {e}")
            continue
        succ = [x for x in w if x["status"] == "SUCCEEDED"]
        exp = [x for x in w if x["status"] == "EXPIRED"]
        print(
            f"==== EP{ep} seed={seed} V={s['total_value']} "
            f"n_succ={s['n_success']} n_exp={s['n_expired']} "
            f"frags={s.get('radio_data_fragments')} remote={s.get('n_remote_stages_done')}"
        )
        print("  SUCCEEDED:", [(x["workflow_id"], x["value"]) for x in succ])
        highs = [
            (x["workflow_id"], x["value"], x["release_s"], x["deadline_s"], x.get("completion_s"))
            for x in w
            if float(x["value"]) >= 12
        ]
        print("  ALL high-value:", highs)
        print("  EXPIRED high:", [(x["workflow_id"], x["value"]) for x in exp if float(x["value"]) >= 12])
        acts = Counter(x["action"] for x in d)
        print("  actions:", dict(acts))
        drops = [x for x in d if x["action"] == "DROP"]
        print("  DROPs:", [(x["workflow_id"], x.get("detail", "")) for x in drops])

    # deeper EP10 timeline vs EP9 for high-value wf
    print("\n==== EP10 decisions on high-value / early arrivals ====")
    _, w10, d10 = load(10, 5110)
    high_ids = {x["workflow_id"] for x in w10 if float(x["value"]) >= 12}
    for x in d10:
        if x["workflow_id"] in high_ids or x["action"] == "DROP":
            print(
                f"  t={x.get('t_s', x.get('time', ''))} {x['action']} wf={x['workflow_id']} "
                f"detail={x.get('detail','')}"
            )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Rewrite live WiLab reward/utility canvas from convergence_history.csv every few seconds."""
from __future__ import annotations

import csv
import time
from pathlib import Path

HIST = Path(__file__).resolve().parents[1] / "outputs" / "wilab_ddqn_convergence" / "convergence_history.csv"
CANVAS = Path(
    r"C:\Users\dash\.cursor\projects\c-Users-dash-Downloads-NewProject\canvases\wilab-reward-utility-live.canvas.tsx"
)
PDP = 45.6
POLL = 8.0


def load_rows() -> list[dict]:
    """CSV only — no fake EP patches (clean curve must stay honest)."""
    if not HIST.is_file():
        return []
    with HIST.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    by_ep = {int(float(r["episode"])): r for r in rows if r.get("episode")}
    return [by_ep[k] for k in sorted(by_ep)]


def render(rows: list[dict]) -> str:
    cats = [str(int(float(r["episode"]))) for r in rows]
    rets = [float(r.get("ep_return") or 0) for r in rows]
    utils = [float(r.get("total_value") or 0) for r in rows]
    mids: list[str] = []
    for r in rows:
        v = r.get("eval_mean_value") or ""
        mids.append("null" if v == "" else f"{float(v):.6g}")
    updated = time.strftime("%Y-%m-%d %H:%M:%S")
    cat_js = ", ".join(f'"{c}"' for c in cats)
    ret_js = ", ".join(f"{v:.6g}" for v in rets)
    util_js = ", ".join(f"{v:.6g}" for v in utils)
    mid_js = ", ".join(mids)
    seeds = [str(int(float(r.get("seed") or 0))) for r in rows]
    seed_js = ", ".join(f'"{s}"' for s in seeds)

    return f'''import {{
  Card, CardBody, CardHeader, Divider, Grid, H1, H2, LineChart,
  Pill, Row, Stack, Stat, Table, Text,
}} from "cursor/canvas";

const CATEGORIES = [{cat_js}] as string[];
const SEEDS = [{seed_js}] as string[];
const REWARD = [{ret_js}] as number[];
const UTILITY = [{util_js}] as number[];
const MID300 = [{mid_js}] as (number | null)[];
const PDP = {PDP};
const UPDATED = "{updated}";

function fillForward(vals: (number | null)[]): number[] {{
  let last = 0;
  return vals.map((v) => {{
    if (v == null) return last;
    last = v;
    return v;
  }});
}}

export default function WilabRewardUtilityLive() {{
  // Do NOT forward-fill mid300 — that fakes a line and confuses vs train utility.
  const midPlot = MID300.map((v) => (v == null ? 0 : v));
  const lastRet = REWARD[REWARD.length - 1] ?? 0;
  const lastUtil = UTILITY[UTILITY.length - 1] ?? 0;
  const bestUtil = Math.max(...UTILITY, 0);
  const lastEp = CATEGORIES[CATEGORIES.length - 1] ?? "—";
  const tableRows = CATEGORIES.map((ep, i) => [
    ep,
    SEEDS[i],
    REWARD[i].toFixed(2),
    UTILITY[i].toFixed(1),
    MID300[i] == null ? "—" : String(MID300[i]),
  ]);

  return (
    <Stack gap={{20}} style={{{{ padding: 20 }}}}>
      <Stack gap={{6}}>
        <Row gap={{10}} align="center">
          <H1>WiLab Mode-2 DDQN — live reward & utility</H1>
          <Pill tone="info">live</Pill>
        </Row>
        <Text tone="secondary" size="small">
          CSV only · {{UPDATED}} · PDP={PDP} · train utility ≠ mid300 ≠ FULL5
        </Text>
      </Stack>

      <Grid columns={{4}} gap={{12}}>
        <Stat value={{lastEp}} label="Last episode" />
        <Stat value={{lastRet.toFixed(2)}} label="Last ep_return (reward)" />
        <Stat
          value={{lastUtil.toFixed(1)}}
          label="Last TRAIN utility (explore)"
          tone={{lastUtil >= PDP ? "success" : "danger"}}
        />
        <Stat value={{bestUtil.toFixed(1)}} label="Best TRAIN utility" tone="success" />
      </Grid>

      <Card>
        <CardHeader>ep_return (reward) — from CSV</CardHeader>
        <CardBody>
          <LineChart
            categories={{CATEGORIES}}
            series={{[{{ name: "ep_return", data: REWARD, tone: "info" }}]}}
            height={{220}}
            beginAtZero
            fill
          />
          <Text tone="secondary" size="small">
            EP6 must be 5.16 if CSV says so — not trainV 48.
          </Text>
        </CardBody>
      </Card>

      <Card>
        <CardHeader>TRAIN total_value (explore) vs mid300 points</CardHeader>
        <CardBody>
          <LineChart
            categories={{CATEGORIES}}
            series={{[
              {{ name: "train_total_value", data: UTILITY, tone: "success" }},
              {{ name: "mid300 (0=no probe)", data: midPlot, tone: "neutral" }},
            ]}}
            height={{240}}
            beginAtZero
            referenceLines={{[{{ value: PDP, label: "PDP 45.6", tone: "danger" }}]}}
          />
          <Text tone="secondary" size="small">
            EP6 train utility = 48 (may sit near a 50 gridline). mid300=72 on probe eps only.
          </Text>
        </CardBody>
      </Card>

      <Divider />
      <H2>Episode log (exact CSV)</H2>
      <Table
        headers={{["EP", "train_seed", "ep_return", "train_V", "mid300"]}}
        rows={{tableRows}}
      />
    </Stack>
  );
}}
'''


def render_waiting() -> str:
    updated = time.strftime("%Y-%m-%d %H:%M:%S")
    return f'''import {{ Card, CardBody, CardHeader, H1, Pill, Row, Stack, Text }} from "cursor/canvas";

export default function WilabRewardUtilityLive() {{
  return (
    <Stack gap={{20}} style={{{{ padding: 20 }}}}>
      <Row gap={{10}} align="center">
        <H1>WiLab Mode-2 DDQN — live reward & utility</H1>
        <Pill tone="warning">waiting</Pill>
      </Row>
      <Text tone="secondary" size="small">Clean shielded train · {{"{updated}"}} · PDP = {PDP}</Text>
      <Card>
        <CardHeader>Status</CardHeader>
        <CardBody>
          <Text>No episodes in convergence_history.csv yet. Charts refresh when EP1 lands.</Text>
        </CardBody>
      </Card>
    </Stack>
  );
}}
'''


def main() -> None:
    last_sig = ""
    print(f"canvas updater → {CANVAS}", flush=True)
    CANVAS.parent.mkdir(parents=True, exist_ok=True)
    while True:
        rows = load_rows()
        if not rows:
            sig = "waiting"
            if sig != last_sig:
                CANVAS.write_text(render_waiting(), encoding="utf-8")
                last_sig = sig
                print("updated waiting placeholder", flush=True)
        else:
            sig = "|".join(
                f"{r.get('episode')}:{r.get('ep_return')}:{r.get('total_value')}:{r.get('eval_mean_value')}"
                for r in rows
            )
            if sig != last_sig:
                CANVAS.write_text(render(rows), encoding="utf-8")
                last_sig = sig
                print(
                    f"updated N={len(rows)} last_ep={rows[-1].get('episode')} "
                    f"ret={rows[-1].get('ep_return')} V={rows[-1].get('total_value')}",
                    flush=True,
                )
        time.sleep(POLL)


if __name__ == "__main__":
    main()

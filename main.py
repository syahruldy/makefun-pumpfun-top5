#!/usr/bin/env python3
"""pumpfun-top5-forecast-01 entry v1e: momentum-shrunk mean-reversion forecast
with a market-drift correction.

Reads /input/history.ndjson (10-min snapshots of new pump.fun coins) and
/input/task.json, writes /output/forecast.json with:
  - probabilities: per-place (1..5) coin probabilities from a Gaussian kernel
    over predicted future log10 market caps, with a large unnamed share for
    coins born after the cut (newcomers take ~1/3 of top-5 slots at 1h and
    more at longer horizons in the practice data).
  - top5: the 5 coins with the highest predicted future log-cap.
  - sizes: per-place log10 market-cap bin distributions (mixture of an
    incumbent normal and a wider newcomer normal around the SAME centre:
    a newcomer holding place k has place k's cap, not #1's).

v1e changes over v1: the top-5 cap level drifts up fast in the practice
data (#1 rose 3.04 -> 3.56 log10 SOL in 40 minutes, and the level was
still rising ~+0.1 per 10 minutes at the cut), so the forecast adds a
market-drift term estimated from the slope of the top-5 level over the
last three marks, capped and saturating over long horizons. The size
spread is tightened (the dev leaders' size scores imply a tighter,
better-centred distribution than v1's).

Horizon H (hours from history cut to the mean answer read) is read from
task.json, so the same code adapts to the 1h practice and the 10h final:
momentum is shrunk and capped (10-min momentum is mean-reverting in the
practice data, corr -0.14), decay grows with H, newcomer share and size
spread grow with H.

Stdlib only, deterministic (no randomness), pure Python.
"""
import json
import math
import os
import sys
from datetime import datetime, timezone

INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")

# ---- tunables (fixed in advance; see module docstring for the evidence) ----
MOM_SHRINK = 0.30        # momentum is mean-reverting at 10-min scale: shrink it hard
MOM_TAU_H = 1.0          # momentum persistence timescale (hours)
MOM_CAP = 0.50           # cap |momentum| at 0.5 log10/hour (10x in 2h)
DECAY_POOL_TOP = 0.02    # log10/hour baseline decay, pool coin in top 20
DECAY_POOL = 0.05        # log10/hour baseline decay, other pool coin
DECAY_CURVE = 0.10       # log10/hour baseline decay, curve/migrating coin
CURVE_CAP_LOG = 2.54     # log10 SOL: bonding-curve coins cap near ~350 SOL
NEWCOMER_BASE = 0.38     # unnamed share at H=0 (2/5 top-5 slots unnameable even at 20-min horizon in backtest)
NEWCOMER_PER_H = 0.03    # extra unnamed share per horizon hour
NEWCOMER_MAX = 0.68      # cap on unnamed share
TAU_BASE = 0.20          # Gaussian kernel width (log10) at H=0
TAU_PER_H = 0.02         # extra kernel width per horizon hour
SIZE_SD_BASE = 0.30      # incumbent size sd (log10) at H=0
SIZE_SD_PER_H = 0.06     # extra size sd per horizon hour
SIZE_SD_MAX = 0.90       # cap on incumbent size sd
NEW_SIZE_SD = 0.45       # newcomer size sd (log10)
NEW_SIZE_BONUS = 0.15    # newcomers reaching top-5 run above the top incumbent
DRIFT_TAU_H = 1.5        # market drift saturates over ~1.5h of horizon
DRIFT_MIN = -0.30        # cap market drift at -0.30 log10/hour
DRIFT_MAX = 0.70         # cap market drift at +0.70 log10/hour
MAX_NAMED_PER_PLACE = 60  # cap on rows per place (limit is 100)
MICRO = 1_000_000        # probabilities rounded down to 6 decimals


def parse_ts(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


def log_cap_sol(market_cap_lamports):
    return math.log10(int(market_cap_lamports)) - 9.0


def norm_cdf(x, mean, sd):
    """P(X <= x) for X ~ Normal(mean, sd)."""
    if sd <= 0:
        return 1.0 if x >= mean else 0.0
    return 0.5 * math.erfc(-(x - mean) / (sd * math.sqrt(2.0)))


def main():
    with open(os.path.join(INPUT_DIR, "task.json")) as f:
        task = json.load(f)
    places = int(task["places"])
    edges = [float(e) for e in task["size"]["edges"]]
    out_name = task.get("output", "forecast.json")

    # ---- horizon: mean answer read minus history cut, in hours ----
    reads = [parse_ts(r) for r in task["reads"]]
    cut = parse_ts(task["history_until"])
    horizon_h = max(0.0, (sum(reads) / len(reads) - cut) / 3600.0)

    # ---- market drift: slope of the top-5 cap level over the last marks ----
    # (filled after marks are parsed; see below)
    drift_total = 0.0

    # ---- parse marks; per-coin series of (t, logcap, venue) ----
    marks = []
    with open(os.path.join(INPUT_DIR, "history.ndjson")) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            m = json.loads(line)
            if m.get("ok") is True and isinstance(m.get("rows"), list) and m["rows"]:
                marks.append(m)
    if not marks:
        raise SystemExit("the history holds no ok mark with rows")
    marks.sort(key=lambda m: parse_ts(m["at"]))
    _lt, _lv = [], []
    for _m in marks[-3:]:
        _rows = sorted(_m["rows"], key=lambda r: r["rank"])[:places]
        _lt.append(parse_ts(_m["at"]))
        _lv.append(sum(log_cap_sol(r["market_cap_lamports"]) for r in _rows) / len(_rows))
    if len(_lt) >= 2:
        _mt = sum(_lt) / len(_lt); _mv = sum(_lv) / len(_lv)
        _num = sum((t - _mt) * (v - _mv) for t, v in zip(_lt, _lv))
        _den = sum((t - _mt) ** 2 for t in _lt)
        _drift = _num / _den * 3600.0 if _den > 0 else 0.0
        _drift = max(DRIFT_MIN, min(DRIFT_MAX, _drift))
        drift_total = _drift * DRIFT_TAU_H * (1.0 - math.exp(-horizon_h / DRIFT_TAU_H)) if horizon_h > 0 else 0.0
    last = marks[-1]
    last_t = parse_ts(last["at"])
    last_rows = sorted(last["rows"], key=lambda r: r["rank"])
    if len(last_rows) < places:
        raise SystemExit("last mark ranks fewer coins than places")

    series = {}   # mint -> list of (t, logcap, venue)
    created = {}  # mint -> created_at timestamp
    for m in marks:
        t = parse_ts(m["at"])
        for r in m["rows"]:
            mint = r["mint"]
            series.setdefault(mint, []).append(
                (t, log_cap_sol(r["market_cap_lamports"]), r.get("venue", "curve")))
            if mint not in created and r.get("created_at"):
                created[mint] = parse_ts(r["created_at"])

    # ---- per-coin prediction at horizon ----
    newcomer = min(NEWCOMER_MAX, NEWCOMER_BASE + NEWCOMER_PER_H * horizon_h)
    tau = TAU_BASE + TAU_PER_H * horizon_h
    size_sd = min(SIZE_SD_MAX, SIZE_SD_BASE + SIZE_SD_PER_H * horizon_h)
    mom_factor = MOM_TAU_H * (1.0 - math.exp(-horizon_h / MOM_TAU_H)) if horizon_h > 0 else 0.0

    preds = []  # (mint, mu, venue, rank_now)
    for rank0, r in enumerate(last_rows):
        mint = r["mint"]
        x_now = log_cap_sol(r["market_cap_lamports"])
        venue = r.get("venue", "curve")
        s = series[mint]

        # momentum: blend of last-30-min and last-2h log-cap slopes (per hour)
        def slope(window_s):
            pts = [(t, x) for (t, x, _v) in s if t >= last_t - window_s]
            if len(pts) < 2:
                return None
            dt_h = (pts[-1][0] - pts[0][0]) / 3600.0
            if dt_h <= 0:
                return None
            return (pts[-1][1] - pts[0][1]) / dt_h

        m30 = slope(30 * 60)
        m2h = slope(2 * 3600)
        moms = [m for m in (m30, m2h) if m is not None]
        mom = sum(moms) / len(moms) if moms else 0.0
        mom = max(-MOM_CAP, min(MOM_CAP, mom))

        decay = DECAY_CURVE
        if venue == "pool":
            decay = DECAY_POOL_TOP if rank0 < 20 else DECAY_POOL

        mu = x_now + drift_total + MOM_SHRINK * mom * mom_factor - decay * horizon_h

        # fresh graduates pump: coins that reached the pool within the last
        # hour get a small boost (practice top coins were all fresh graduates).
        if venue == "pool":
            first_pool_t = None
            for (t, _x, v) in s:
                if v == "pool":
                    first_pool_t = t
                    break
            if first_pool_t is not None:
                age_grad_h = (last_t - first_pool_t) / 3600.0
                if age_grad_h < 1.0:
                    mu += 0.10 * (1.0 - age_grad_h)

        # curve coins cannot exceed the bonding-curve cap unless they graduate;
        # a fast-growing coin near completion gets partial graduation upside.
        if venue != "pool":
            grad = 0.0
            if x_now > 2.0 and mom > 0:
                grad = min(1.0, (10 ** x_now / 350.0)) * min(1.0, mom / 0.3)
            mu = min(mu, CURVE_CAP_LOG + 0.5 * grad)

        preds.append((mint, mu, venue, rank0 + 1))

    preds.sort(key=lambda p: -p[1])
    mu_top = [p[1] for p in preds[:places]]
    # fill if fewer predicted than places (should not happen)
    while len(mu_top) < places:
        mu_top.append(mu_top[-1] if mu_top else 2.0)

    # ---- probabilities: Gaussian kernel around each place's predicted cap ----
    named_mass = 1.0 - newcomer
    probabilities = []
    for k in range(1, places + 1):
        centre = mu_top[k - 1]
        weights = []
        for (mint, mu, _venue, _rank) in preds:
            d = (mu - centre) / tau
            if abs(d) > 4.0:
                continue
            weights.append((mint, math.exp(-0.5 * d * d)))
        weights.sort(key=lambda w: -w[1])
        weights = weights[:MAX_NAMED_PER_PLACE]
        total = sum(w for _, w in weights)
        if total <= 0:
            continue
        for mint, w in weights:
            p = math.floor(named_mass * w / total * MICRO) / MICRO
            if p > 0:
                probabilities.append({"mint": mint, "place": k, "p": p})

    top5 = [p[0] for p in preds[:places]]

    # ---- sizes: mixture of incumbent + newcomer normals per place ----
    # newcomer centre per place: a newcomer holding place k has place k's
    # cap, not #1's (the old #1-centred mixture miscentred places 2-5
    # badly). Same centre as incumbents, wider spread.
    sizes = []
    nbins = len(edges) + 1  # bins 0..len(edges)
    for k in range(1, places + 1):
        mu_inc = mu_top[k - 1]
        for b in range(nbins):
            lo = -math.inf if b == 0 else edges[b - 1]
            hi = math.inf if b == len(edges) else edges[b]
            p_inc = norm_cdf(hi, mu_inc, size_sd) - norm_cdf(lo, mu_inc, size_sd)
            p_new = norm_cdf(hi, mu_inc, NEW_SIZE_SD) - norm_cdf(lo, mu_inc, NEW_SIZE_SD)
            p = (1.0 - newcomer) * p_inc + newcomer * p_new
            u = math.floor(p * MICRO)
            if u > 0:
                sizes.append({"place": k, "bin": b, "p": u / MICRO})

    forecast = {"probabilities": probabilities, "top5": top5, "sizes": sizes}
    out_path = os.path.join(OUTPUT_DIR, out_name)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(out_path, "w") as f:
        f.write(json.dumps(forecast) + "\n")
    print(f"wrote {out_path}: horizon={horizon_h:.2f}h drift_total={drift_total:.2f} newcomer={newcomer:.2f} "
          f"tau={tau:.2f} size_sd={size_sd:.2f} probs={len(probabilities)} "
          f"sizes={len(sizes)} top5={[p[0][:8] for p in preds[:places]]}",
          file=sys.stderr)


if __name__ == "__main__":
    main()

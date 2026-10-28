"""
Build the exception console from the paper's own code.

1. Runs src/experiment.py unchanged and checks every number in the committed
   results/results.json is reproduced exactly (fails the build otherwise).
2. Exports the model, test set, severity and tiers for the console, adding
   per-shipment risk drivers from the forest's decision paths.
3. Writes the static site to _site/.

Everything model-related comes from experiment.py. The console adds a snapshot
time, workflow status and shipment IDs so the queue looks like a working day;
those parts are illustrative and labelled as such in the UI.
"""
import json
import runpy
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
published = json.loads((ROOT / "results" / "results.json").read_text())
g = runpy.run_path(str(ROOT / "src" / "experiment.py"), run_name="__main__")
reproduced = json.loads((ROOT / "results" / "results.json").read_text())


def flat(d, p=""):
    out = {}
    for k, v in d.items():
        out.update(flat(v, p + k + ".") if isinstance(v, dict) else {p + k: v})
    return out


fp, fr = flat(published), flat(reproduced)
mismatch = [k for k in fp if fp[k] != fr.get(k)]
print(f"Reproducibility check: {len(fp) - len(mismatch)} of {len(fp)} published results match")
if mismatch:
    print("Mismatched:", mismatch[:10])
    sys.exit(1)

S = {k: g[k] for k in ["df", "X", "T", "rf", "OUT", "thr_rf", "S1", "S2", "v95", "lane_med", "test", "rule"]}
df, X, T, rf, OUT = S["df"], S["X"], S["T"].copy(), S["rf"], S["OUT"]
thr, S1, S2, v95, lane_med = S["thr_rf"], S["S1"], S["S2"], S["v95"], S["lane_med"]
test = S["test"]
SITE = ROOT / "_site"; SITE.mkdir(exist_ok=True)
OUTDIR = SITE

NOW = pd.Timestamp("2025-10-14 10:00")
T["pred_time"] = pd.to_datetime(T.pred_time)
T["planned_arrival"] = pd.to_datetime(T.planned_arrival)
T["ship_ts"] = pd.to_datetime(T.ship_ts)
T["flag_time"] = T.pred_time.dt.ceil("4h")                     # agent runs in 4-hour batches (paper)

# severity components, exactly as in the paper
T["c_inv"] = np.clip(T.exp_delay_days / np.maximum(T.days_supply, 0.5), 0, 1)
T["c_sla"] = np.where(T.sla_linked == 1, 1.0, 0.2)
T["c_val"] = np.clip(T.order_value / v95, 0, 1)
assert np.allclose(T.S, T.p * (0.5 * T.c_inv + 0.3 * T.c_sla + 0.2 * T.c_val))

# ---------------------------------------------------------------- snapshot
in_transit = (T.pred_time <= NOW) & (T.planned_arrival > NOW)
snap = T[in_transit].copy()
snap["sid"] = ["SHP-" + str(400000 + int(i)) for i in snap.index]
flagged = snap[snap.flag].copy()

# per-shipment risk drivers: tree-path contributions of the Random Forest (Saabas method)
Xs = X.loc[flagged.index]
cols = list(X.columns)
group = [c.split("_")[0] if c.split("_")[0] in ["mode", "origin", "dest", "carrier"] else c for c in cols]
contrib = np.zeros((len(Xs), len(cols)))
bias = np.zeros(len(Xs))
Xv = Xs.values.astype(np.float32)
for est in rf.estimators_:
    t = est.tree_
    val = t.value[:, 0, :]
    prob = val[:, 1] / val.sum(axis=1)
    paths = est.decision_path(Xv)
    for r in range(len(Xs)):
        nodes = paths.indices[paths.indptr[r]:paths.indptr[r + 1]]
        bias[r] += prob[nodes[0]]
        for a, b in zip(nodes[:-1], nodes[1:]):
            contrib[r, t.feature[a]] += prob[b] - prob[a]
n_est = len(rf.estimators_)
contrib /= n_est; bias /= n_est
p_check = bias + contrib.sum(axis=1)
assert np.allclose(p_check, flagged.p.values, atol=1e-6), np.abs(p_check - flagged.p.values).max()
gnames = sorted(set(group))
G = np.zeros((len(Xs), len(gnames)))
for j, g in enumerate(group):
    G[:, gnames.index(g)] += contrib[:, j]

LABEL = {
    "late_ms_hours": "Late at last milestone", "recent_delay_freq": "Carrier late rate, last 30 days",
    "hist_ontime": "Carrier on-time history", "mode": "Transport mode", "customs": "Customs hold",
    "carrier": "Carrier", "weather": "Weather severity", "lane_hist_var": "Lane variability",
    "holiday": "Holiday period", "origin": "Origin", "dest": "Destination DC", "planned_days": "Planned transit",
    "weight_t": "Shipment weight", "week": "Week of year", "month": "Month", "dow": "Ship day of week",
}


def shown_value(g, row):
    return {
        "late_ms_hours": f"{row.late_ms_hours:.0f} h", "recent_delay_freq": f"{row.recent_delay_freq:.0%}",
        "hist_ontime": f"{row.hist_ontime:.0%}", "mode": row["mode"], "customs": "Yes" if row.customs else "No",
        "carrier": row.carrier, "weather": f"{row.weather:.2f}", "lane_hist_var": f"{row.lane_hist_var:.2f}",
        "holiday": "Yes" if row.holiday else "No", "origin": row.origin, "dest": row.dest,
        "planned_days": f"{row.planned_days} days", "weight_t": f"{row.weight_t:.1f} t", "week": str(row.week),
        "month": str(row.month), "dow": ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][row.dow],
    }[g]


# illustrative Tier 2 proposal: follows the largest severity driver (console rule, not part of the paper)
def proposal(row):
    parts = {"inv": 0.5 * row.c_inv, "sla": 0.3 * row.c_sla, "val": 0.2 * row.c_val}
    k = max(parts, key=parts.get)
    if k == "inv":
        return "Reallocate stock from another DC to cover the gap"
    if k == "sla":
        return "Expedite final leg to protect the customer SLA" if row["mode"] == "Ocean" else "Upgrade to priority air service"
    return "Reroute to the fastest available lane"


def status(row):
    age_h = (NOW - row.flag_time).total_seconds() / 3600
    if row.tier == 1:
        return "auto", "ETA updated in TMS, stakeholders notified"
    if row.tier == 2:
        if age_h < 24:
            return "awaiting", "Awaiting planner approval"
        return ("approved", "Approved by planner") if row.y == 1 else ("rejected", "Rejected by planner")
    if age_h < 72:
        return "escalated", "Escalated, planner leading"
    return "resolved", "Resolved by planner"


ships = []
for k, (idx, row) in enumerate(flagged.iterrows()):
    order = np.argsort(-G[k])
    drivers = [{"f": LABEL[gnames[j]], "v": shown_value(gnames[j], row), "c": round(float(G[k, j]), 4)}
               for j in order[:5] if G[k, j] > 0.004]
    st, st_text = status(row)
    exp_arr = row.planned_arrival + pd.Timedelta(days=float(row.exp_delay_days))
    ships.append({
        "id": row.sid, "mode": row["mode"], "origin": row.origin, "dest": row.dest, "carrier": row.carrier,
        "shipped": row.ship_ts.strftime("%Y-%m-%d"), "predAt": row.pred_time.strftime("%Y-%m-%d %H:%M"),
        "flagAt": row.flag_time.strftime("%Y-%m-%d %H:%M"), "planned": row.planned_arrival.strftime("%Y-%m-%d"),
        "expected": exp_arr.strftime("%Y-%m-%d"), "plannedDays": int(row.planned_days),
        "p": round(float(row.p), 3), "expDelayDays": round(float(row.exp_delay_days), 1),
        "daysSupply": round(float(row.days_supply), 1), "sla": int(row.sla_linked), "value": round(float(row.order_value)),
        "cInv": round(float(row.c_inv), 3), "cSla": round(float(row.c_sla), 3), "cVal": round(float(row.c_val), 3),
        "S": round(float(row.S), 3), "tier": int(row.tier), "status": st, "statusText": st_text,
        "proposal": proposal(row) if row.tier == 2 else None,
        "base": round(float(bias[k]), 3), "drivers": drivers,
        "lateMs": round(float(row.late_ms_hours)), "custom": int(row.customs), "weather": round(float(row.weather), 2),
    })

# ------------------------------------------------- arrays for the policy page
TT = T.copy()
policy = {k: [round(float(v), 3) for v in TT[c]] for k, c in
          [("p", "p"), ("cInv", "c_inv"), ("cSla", "c_sla"), ("cVal", "c_val")]}
policy["y"] = [int(v) for v in TT.y]
policy["rule"] = [int(v) for v in S["rule"]]
# exp_delay_days depends on p; c_inv was computed with the paper's p, so it is exported as is
# check: expected planner hours from the paper's mean handling times
H = {"investigate": 3.0, "fp": 1.0, "approve": 0.5, "reject": 0.3}
fl, y, tier = TT.flag.values, TT.y.values, TT.tier.values
agent_h = (H["investigate"] * ((fl & (tier == 3) & (y == 1)) | (~fl & (y == 1))).sum()
           + H["fp"] * (fl & (tier == 3) & (y == 0)).sum()
           + H["approve"] * (fl & (tier == 2) & (y == 1)).sum() + H["reject"] * (fl & (tier == 2) & (y == 0)).sum())
rule = np.array(policy["rule"]).astype(bool)
manual_h = H["investigate"] * (y == 1).sum() + H["fp"] * (rule & (y == 0)).sum()
print(f"expected planner hours  agent {agent_h:.0f} (sim {OUT['sim_base']['agent']['planner_hours'][0]:.0f})"
      f"  manual {manual_h:.0f} (sim {OUT['sim_base']['manual']['planner_hours'][0]:.0f})")

# -------------------------------------------------------- model diagnostics
yt, p = TT.y.values, TT.p.values
bins = np.linspace(0, 1, 11); idx = np.clip(np.digitize(p, bins) - 1, 0, 9)
calib = [{"lo": float(bins[i]), "hi": float(bins[i + 1]), "n": int((idx == i).sum()),
          "pred": float(p[idx == i].mean()) if (idx == i).any() else None,
          "obs": float(yt[idx == i].mean()) if (idx == i).any() else None} for i in range(10)]
imp = pd.Series(rf.feature_importances_, index=cols)
gimp = {}
for c, v in imp.items():
    g = c.split("_")[0] if c.split("_")[0] in ["mode", "origin", "dest", "carrier"] else c
    gimp[LABEL[g]] = gimp.get(LABEL[g], 0) + float(v)
gimp = dict(sorted(gimp.items(), key=lambda kv: -kv[1]))
pred = p >= thr
conf = {"tp": int((pred & (yt == 1)).sum()), "fp": int((pred & (yt == 0)).sum()),
        "fn": int((~pred & (yt == 1)).sum()), "tn": int((~pred & (yt == 0)).sum())}

snap_counts = {
    "inTransit": int(in_transit.sum()), "flagged": int(len(flagged)),
    "monitoring": int(in_transit.sum() - len(flagged)),
    "byTier": {str(k): int((flagged.tier == k).sum()) for k in [1, 2, 3]},
    "valueAtRisk": round(float((flagged.order_value * flagged.p).sum())),
    "new24": int((flagged.flag_time > NOW - pd.Timedelta("24h")).sum()),
    "medianLeadDays": round(float(((flagged.planned_arrival - NOW).dt.total_seconds() / 86400).median()), 1),
}

data = {
    "now": NOW.strftime("%Y-%m-%d %H:%M"), "threshold": thr, "S1": S1, "S2": S2,
    "weights": {"inv": 0.5, "sla": 0.3, "val": 0.2}, "snapshot": snap_counts, "shipments": ships,
    "policy": policy, "handling": H,
    "paper": {k: OUT[k] for k in ["n", "delay_rate", "n_carriers", "n_lanes", "ocean_share", "n_train", "n_val", "n_test",
                                  "thr_rf", "thr_lr", "metrics", "importance_top", "tier_share", "sim_base", "mttr_reduction",
                                  "planner_hours_reduction", "sensitivity", "recall_early_rule", "recall_early_agent",
                                  "lead_hours_median", "n_test_delays", "rf_params"]},
    "importance": gimp, "calibration": calib, "confusion": conf,
    "expectedHours": {"agent": round(float(agent_h)), "manual": round(float(manual_h))},
}
(OUTDIR / "data.js").write_text("window.DRA = " + json.dumps(data, separators=(",", ":")) + ";\n")
print("shipments", len(ships), "bytes", (OUTDIR / "data.js").stat().st_size)
print("status mix", pd.Series([s["status"] for s in ships]).value_counts().to_dict())
print("snapshot", snap_counts)
page = (ROOT / "console" / "index.html").read_text()
head_end = page.index("</style>") + len("</style>")
(SITE / "index.html").write_text('<!doctype html>\n<html lang="en">\n<head>\n' + page[:head_end] + "\n</head>\n<body>\n" + page[head_end:] + "\n</body>\n</html>\n")
shutil.copy(ROOT / "figures" / "architecture.png", SITE / "architecture.png")
(SITE / ".nojekyll").write_text("")
print("site written to _site/")

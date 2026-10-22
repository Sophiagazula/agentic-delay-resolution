"""Synthetic shipment data, delay models, and workflow simulation for the paper.
All numbers reported in the paper are produced by this script (seed fixed).
Run from anywhere:  python src/experiment.py   (about 3-5 minutes)
Outputs: data/synthetic_shipments.csv and results/results.json"""
import json
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
(ROOT / "data").mkdir(exist_ok=True); (ROOT / "results").mkdir(exist_ok=True)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score

rng = np.random.default_rng(42)
OUT = {}

# ---------------- 1. Synthetic data ----------------
N = 48000
start = pd.Timestamp("2024-01-01")
ship_day = np.sort(rng.integers(0, 731, N))
ship_ts = start + pd.to_timedelta(ship_day, "D") + pd.to_timedelta(rng.integers(0, 24, N), "h")

origins = ["China", "Vietnam", "India", "Germany", "Japan", "South Korea", "Taiwan", "Thailand"]
origin_w = np.array([.32, .14, .12, .10, .09, .09, .08, .06])
dests = ["DC-Midwest", "DC-South", "DC-East", "DC-West"]
lanes = [(o, d) for o in origins for d in dests]  # 32 lanes
lane_var = {l: rng.uniform(0.2, 1.0) for l in lanes}
base_transit = {"China": 32, "Vietnam": 34, "India": 38, "Germany": 26, "Japan": 28, "South Korea": 29, "Taiwan": 31, "Thailand": 35}
customs_risk = {"China": .07, "Vietnam": .06, "India": .08, "Germany": .02, "Japan": .02, "South Korea": .03, "Taiwan": .04, "Thailand": .05}

ocean_carriers = [f"OC{i}" for i in range(1, 9)]
air_carriers = [f"AC{i}" for i in range(1, 5)]
rel = {c: rng.beta(8, 2.5) for c in ocean_carriers + air_carriers}

mode = np.where(rng.random(N) < 0.72, "Ocean", "Air")
origin = rng.choice(origins, N, p=origin_w)
dest = rng.choice(dests, N)
carrier = np.where(mode == "Ocean", rng.choice(ocean_carriers, N), rng.choice(air_carriers, N))
planned_days = np.where(mode == "Ocean",
                        np.array([base_transit[o] for o in origin]) + rng.integers(-2, 4, N),
                        rng.integers(3, 8, N))
weight_t = np.where(mode == "Ocean", rng.lognormal(2.3, 0.5, N), rng.lognormal(0.0, 0.6, N))

woy = ship_ts.isocalendar().week.values.astype(int)
month = ship_ts.month.values
dow = ship_ts.dayofweek.values
doy = ship_ts.dayofyear.values
# holiday proximity: Lunar New Year (~late Jan/Feb), Golden Week (early Oct), year-end
def near(d, centers, w=14):
    return np.min([np.abs(d - c) for c in centers], axis=0) <= w
holiday = near(doy, [40, 278, 355]).astype(int)
peak = np.isin(month, [8, 9, 10, 11]).astype(int)
asia = np.isin(origin, ["China", "Vietnam", "Japan", "South Korea", "Taiwan", "Thailand"])
weather = np.clip(rng.beta(2, 6, N) + 0.25 * (asia & np.isin(month, [7, 8, 9, 10])) + 0.15 * np.isin(month, [12, 1, 2]), 0, 1)
customs = (rng.random(N) < np.array([customs_risk[o] for o in origin]) * (1 + 0.8 * holiday)).astype(int)

# carrier congestion shocks (AR(1) per carrier per day)
shock = {}
for c in rel:
    s = np.zeros(731);
    for t in range(1, 731): s[t] = 0.93 * s[t - 1] + rng.normal(0, 0.25)
    shock[c] = s
cong = np.array([shock[c][d] for c, d in zip(carrier, ship_day)])
recent_delay_freq = np.clip(0.15 + 0.12 * cong + rng.normal(0, 0.05, N), 0, 1)  # observed rolling 30-day late share
hist_ontime = np.clip(np.array([rel[c] for c in carrier]) + rng.normal(0, 0.03, N), 0, 1)
lv = np.array([lane_var[(o, d)] for o, d in zip(origin, dest)])
lane_hist_var = np.clip(lv + rng.normal(0, 0.08, N), 0, None)

latent = (-2.35 + 3.0 * (1 - np.array([rel[c] for c in carrier])) + 0.9 * lv + 1.4 * weather + 1.6 * customs
          + 0.8 * cong + 0.45 * holiday + 0.35 * peak + 0.3 * (mode == "Ocean") + rng.normal(0, 0.9, N))
delay_hours = np.where(latent > 0, np.exp(3.2 + 0.55 * latent + rng.normal(0, 0.35, N)), rng.exponential(6, N))
delay_hours *= np.where(mode == "Air", 0.45, 1.0)
y = (delay_hours > 48).astype(int)

# milestone signal observed at prediction time (~40% of planned transit)
late_ms_hours = np.clip(0.35 * delay_hours * (rng.random(N) < 0.6) + rng.normal(0, 10, N), 0, None)
carrier_eta_slip = 0.55 * delay_hours * (rng.random(N) < 0.55) + rng.normal(0, 14, N)

# business context
days_supply = rng.gamma(2.2, 4.0, N)
sla_linked = (rng.random(N) < 0.35).astype(int)
order_value = rng.lognormal(10.2, 0.9, N)

df = pd.DataFrame(dict(ship_ts=ship_ts, mode=mode, origin=origin, dest=dest, carrier=carrier, lane=[f"{o}>{d}" for o, d in zip(origin, dest)],
    planned_days=planned_days, weight_t=weight_t, week=woy, month=month, dow=dow, holiday=holiday, weather=weather, customs=customs,
    hist_ontime=hist_ontime, recent_delay_freq=recent_delay_freq, lane_hist_var=lane_hist_var, late_ms_hours=late_ms_hours,
    carrier_eta_slip=carrier_eta_slip, days_supply=days_supply, sla_linked=sla_linked, order_value=order_value,
    delay_hours=delay_hours, y=y))
df["planned_arrival"] = df.ship_ts + pd.to_timedelta(df.planned_days, "D")
df["pred_time"] = df.ship_ts + pd.to_timedelta(0.4 * df.planned_days, "D")
df.to_csv(ROOT / "data" / "synthetic_shipments.csv", index=False)

OUT["n"] = int(N); OUT["delay_rate"] = float(y.mean()); OUT["n_carriers"] = len(rel); OUT["n_lanes"] = len(lanes)
OUT["ocean_share"] = float((mode == "Ocean").mean())

# ---------------- 2. Models ----------------
num = ["planned_days", "weight_t", "week", "month", "dow", "holiday", "weather", "customs", "hist_ontime", "recent_delay_freq", "lane_hist_var", "late_ms_hours"]
X = pd.get_dummies(df[num + ["mode", "origin", "dest", "carrier"]], columns=["mode", "origin", "dest", "carrier"], dtype=float)
train = df.ship_ts < "2025-04-01"; val = (df.ship_ts >= "2025-04-01") & (df.ship_ts < "2025-07-01"); test = df.ship_ts >= "2025-07-01"
trval = train | val
OUT["n_train"] = int(train.sum()); OUT["n_val"] = int(val.sum()); OUT["n_test"] = int(test.sum())

grid = GridSearchCV(RandomForestClassifier(n_estimators=300, class_weight="balanced", n_jobs=-1, random_state=0),
                    {"max_depth": [8, 14, None], "min_samples_leaf": [5, 20]}, cv=TimeSeriesSplit(3), scoring="average_precision", n_jobs=1)
grid.fit(X[train], y[train]); OUT["rf_params"] = grid.best_params_
rf = grid.best_estimator_
lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")).fit(X[train], y[train])

def best_thr(p, yy):
    ts = np.linspace(0.1, 0.9, 81); f = [f1_score(yy, p >= t) for t in ts]; return float(ts[int(np.argmax(f))])
thr_rf = best_thr(rf.predict_proba(X[val])[:, 1], y[val]); thr_lr = best_thr(lr.predict_proba(X[val])[:, 1], y[val])
OUT["thr_rf"] = thr_rf; OUT["thr_lr"] = thr_lr
# refit on train+val with chosen hyperparameters
rf.fit(X[trval], y[trval]); lr.fit(X[trval], y[trval])

yt = y[test]
p_rf = rf.predict_proba(X[test])[:, 1]; p_lr = lr.predict_proba(X[test])[:, 1]
rule = (df.carrier_eta_slip[test] > 48).values
def m(pred, score=None):
    d = dict(precision=precision_score(yt, pred), recall=recall_score(yt, pred), f1=f1_score(yt, pred))
    if score is not None: d.update(roc_auc=roc_auc_score(yt, score), pr_auc=average_precision_score(yt, score))
    return {k: round(float(v), 3) for k, v in d.items()}
OUT["metrics"] = {"rule": m(rule), "lr": m(p_lr >= thr_lr, p_lr), "rf": m(p_rf >= thr_rf, p_rf)}
imp = pd.Series(rf.feature_importances_, index=X.columns)
grp = {}
for c, v in imp.items():
    g = c.split("_")[0] if c.split("_")[0] in ["mode", "origin", "dest", "carrier"] else c
    grp[g] = grp.get(g, 0) + v
OUT["importance_top"] = {k: round(float(v), 3) for k, v in sorted(grp.items(), key=lambda kv: -kv[1])[:6]}

# ---------------- 3. Severity + tiers ----------------
T = df[test].copy(); T["p"] = p_rf; T["flag"] = T.p >= thr_rf
lane_med = df[trval & (df.y == 1)].groupby("lane").delay_hours.median()
T["exp_delay_days"] = T.p * T.lane.map(lane_med).fillna(lane_med.median()) / 24
c_inv = np.clip(T.exp_delay_days / np.maximum(T.days_supply, 0.5), 0, 1)
c_sla = np.where(T.sla_linked == 1, 1.0, 0.2)
v95 = df[trval].order_value.quantile(0.95); c_val = np.clip(T.order_value / v95, 0, 1)
T["S"] = T.p * (0.5 * c_inv + 0.3 * c_sla + 0.2 * c_val)
S1, S2 = 0.20, 0.40
T["tier"] = np.where(~T.flag, 0, np.where(T.S < S1, 1, np.where(T.S < S2, 2, 3)))
fl = T[T.flag]
OUT["tier_share"] = {f"t{k}": round(float((fl.tier == k).mean()), 3) for k in [1, 2, 3]}
OUT["S1"], OUT["S2"] = S1, S2

# ---------------- 4. Workflow simulation ----------------
def next_work(t):
    t = pd.Timestamp(t)
    while True:
        if t.dayofweek >= 5: t = (t + pd.Timedelta(days=7 - t.dayofweek)).normalize() + pd.Timedelta(hours=8); continue
        if t.hour < 8: t = t.normalize() + pd.Timedelta(hours=8)
        if t.hour >= 18: t = t.normalize() + pd.Timedelta(days=1, hours=8); continue
        return t
def add_work(t, h):
    t = next_work(t)
    while h > 1e-9:
        end = t.normalize() + pd.Timedelta(hours=18)
        avail = (end - t).total_seconds() / 3600
        if h <= avail: return t + pd.Timedelta(hours=h)
        h -= avail; t = next_work(end)
    return t

def ln(mean, cv, r, n):
    s = np.sqrt(np.log(1 + cv ** 2)); return r.lognormal(np.log(mean) - s ** 2 / 2, s, n)

def simulate(workflow, planners=8, hmult=1.0, seed=0):
    r = np.random.default_rng(seed)
    tasks = []  # (create, planner_hours or None for automated, is_true_delay)
    D = T
    if workflow == "manual":
        flagged = rule
        for i, (pt, pa, yy, fg) in enumerate(zip(D.pred_time, D.planned_arrival, D.y, flagged)):
            if fg: tasks.append((pt, "investigate" if yy else "fp", yy))
            elif yy: tasks.append((pa, "investigate", yy))  # caught when planned arrival is missed
    else:
        for pt, pa, yy, fg, tier in zip(D.pred_time, D.planned_arrival, D.y, D.flag, D.tier):
            # agent runs in 4-hour batches
            ct = pd.Timestamp(pt).ceil("4h")
            if fg:
                if tier == 1: tasks.append((ct, "auto", yy))
                elif tier == 2: tasks.append((ct, "approve" if yy else "reject", yy))
                else: tasks.append((ct, "investigate" if yy else "fp", yy))
            elif yy: tasks.append((pa, "investigate", yy))
    tasks.sort(key=lambda x: x[0])
    kinds = np.array([k for _, k, _ in tasks])
    dur = np.zeros(len(tasks))
    for k, (mu, cv) in {"investigate": (3.0, 0.5), "fp": (1.0, 0.5), "approve": (0.5, 0.5), "reject": (0.3, 0.5)}.items():
        idx = kinds == k; dur[idx] = ln(mu * hmult, cv, r, idx.sum())
    free = [pd.Timestamp("2025-01-01")] * planners
    res = []; phours = 0.0
    for (ct, k, yy), d in zip(tasks, dur):
        if k == "auto":
            res.append(((r.uniform(5, 20)) / 60, yy)); continue
        j = int(np.argmin(free)); st = next_work(max(ct, free[j])); en = add_work(st, d)
        free[j] = en; phours += d
        res.append(((en - ct).total_seconds() / 3600, yy))
    ttr = np.array([a for a, _ in res]); tru = np.array([b for _, b in res]) == 1
    return dict(n_exc=len(tasks), mttr_true=ttr[tru].mean(), median_true=np.median(ttr[tru]), mttr_all=ttr.mean(), planner_hours=phours,
                share_auto=float((kinds == "auto").mean()))

def reps(wf, **kw):
    rs = [simulate(wf, seed=s, **kw) for s in range(10)]
    return {k: (float(np.mean([x[k] for x in rs])), float(np.std([x[k] for x in rs]))) for k in rs[0]}

base = {"manual": reps("manual"), "agent": reps("agent")}
OUT["sim_base"] = base
OUT["mttr_reduction"] = 1 - base["agent"]["mttr_true"][0] / base["manual"]["mttr_true"][0]
OUT["planner_hours_reduction"] = 1 - base["agent"]["planner_hours"][0] / base["manual"]["planner_hours"][0]
sens = {}
for pl in [7, 8, 10]:
    for hm in [0.7, 1.0, 1.3]:
        a = reps("agent", planners=pl, hmult=hm); b = reps("manual", planners=pl, hmult=hm)
        sens[f"{pl}_{hm}"] = dict(manual=round(b["mttr_true"][0], 2), agent=round(a["mttr_true"][0], 2), red=round(1 - a["mttr_true"][0] / b["mttr_true"][0], 3))
OUT["sensitivity"] = sens

# detection lead time before planned arrival (true delays that were flagged early)
tr = T.y == 1
lead_rule = ((T.planned_arrival - T.pred_time).dt.total_seconds() / 3600)[tr & rule].median()
OUT["recall_early_rule"] = float((tr & rule).sum() / tr.sum()); OUT["recall_early_agent"] = float((tr & T.flag).sum() / tr.sum())
OUT["lead_hours_median"] = float(lead_rule)
OUT["n_test_delays"] = int(tr.sum())

json.dump(OUT, open(ROOT / "results" / "results.json", "w"), indent=1, default=str)
print(json.dumps(OUT, indent=1, default=str))

"""Step 1: generate workload scenarios, simulate every candidate policy on each (identical
config), save raw CSVs: features, per-policy metrics, and the best-policy label."""
import os, sys, time
import numpy as np, pandas as pd
from multiprocessing import Pool
import config as C
from simulator import simulate
from workload_generator import generate_scenario, extract_features


def cost(m):
    return sum(w * m[k] for k, w in C.COST_WEIGHTS.items())


def run_one(args):
    sid, n, seed = args
    rng = np.random.default_rng(seed)
    procs, wtype = generate_scenario(n, rng)
    feat = extract_features(procs)
    rows = []
    for name, (algo, q) in C.POLICIES.items():
        m = simulate(procs, algo, q, C.CS_COST)
        rows.append({"scenario_id": sid, "size": n, "policy": name, **m, "cost": cost(m)})
    return {"scenario_id": sid, "size": n, "wtype": wtype, **feat}, rows


def main(out="data/raw"):
    os.makedirs(out, exist_ok=True)
    jobs, sid = [], 0
    for n in C.SIZES:
        for _ in range(C.SCENARIOS_PER_SIZE):
            jobs.append((sid, n, C.SEED * 100003 + sid)); sid += 1
    t0 = time.time()
    with Pool(max(1, os.cpu_count() or 1)) as p:
        res = p.map(run_one, jobs, chunksize=4)
    feats = pd.DataFrame([r[0] for r in res])
    metrics = pd.DataFrame([row for r in res for row in r[1]])
    lab = metrics[metrics.policy.isin(C.LABEL_POLICIES)]
    best = lab.loc[lab.groupby("scenario_id")["cost"].idxmin(), ["scenario_id", "policy"]]
    feats = feats.merge(best.rename(columns={"policy": "best_policy"}), on="scenario_id")
    feats.to_csv(f"{out}/scenario_features.csv", index=False)
    metrics.to_csv(f"{out}/policy_metrics_long.csv", index=False)
    print(f"{len(feats)} scenarios, {len(metrics)} simulation rows in {time.time()-t0:.0f}s")
    print(feats.groupby("size")["best_policy"].value_counts().unstack(fill_value=0))


if __name__ == "__main__":
    main()

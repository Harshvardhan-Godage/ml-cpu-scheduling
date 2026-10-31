"""Step 2: train Decision Tree, Random Forest and a Logistic-Regression baseline to predict the best
scheduling policy / time quantum from workload features; evaluate classification accuracy AND the
resulting OS-level (simulated) scheduling performance; run statistical tests and make plots."""
import os, json
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import config as C

RAW, RES, FIG = "data/raw", "results", "results/figures"
os.makedirs(FIG, exist_ok=True)
MODELS = {
    "DecisionTree": lambda s: DecisionTreeClassifier(max_depth=6, min_samples_leaf=3, random_state=s),
    "RandomForest": lambda s: RandomForestClassifier(n_estimators=200, min_samples_leaf=2, random_state=s, n_jobs=-1),
    "LogReg(baseline)": lambda s: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
}

feats = pd.read_csv(f"{RAW}/scenario_features.csv").set_index("scenario_id")
met = pd.read_csv(f"{RAW}/policy_metrics_long.csv")
M = {k: met.pivot(index="scenario_id", columns="policy", values=k) for k in ["cost", "avg_wait", "avg_resp", "p95_wait", "avg_turn", "ctx_switches"]}
FEAT_COLS = [c for c in feats.columns if c not in ("size", "wtype", "best_policy")]
oracle_label_cost = M["cost"][C.LABEL_POLICIES].min(axis=1)


def lookup(mat, ids, choice):
    return np.array([mat.loc[i, c] for i, c in zip(ids, choice)])


def method_table(ids, choices):
    """choices: dict method -> array of chosen policy names. Returns per-scenario DataFrame per method."""
    out = {}
    for m, ch in choices.items():
        d = {k: lookup(M[k], ids, ch) for k in ["cost", "avg_wait", "avg_resp", "p95_wait", "ctx_switches"]}
        d["regret_pct"] = 100 * (d["cost"] - oracle_label_cost.loc[ids].values) / oracle_label_cost.loc[ids].values
        out[m] = pd.DataFrame(d, index=ids)
    return out


def boot_ci(x, n=2000, seed=0):
    r = np.random.default_rng(seed)
    b = [r.choice(x, len(x)).mean() for _ in range(n)]
    return np.percentile(b, [2.5, 97.5])


summary, accrows, stat_rows, oof_store, box_data = [], [], [], {}, {}
for size in C.SIZES:
    d = feats[feats["size"] == size]
    ids, X, y = d.index.values, d[FEAT_COLS].values, d["best_policy"].values
    preds = {m: [] for m in MODELS}
    for r in range(C.CV_REPEATS):
        for m in MODELS:
            p = np.empty(len(y), dtype=object)
            for k, (tr, te) in enumerate(KFold(C.CV_FOLDS, shuffle=True, random_state=C.SEED + r).split(X)):
                clf = MODELS[m](C.SEED + r).fit(X[tr], y[tr])
                p[te] = clf.predict(X[te])
                accrows.append({"size": size, "model": m, "repeat": r, "fold": k,
                                "accuracy": accuracy_score(y[te], p[te]),
                                "macro_f1": f1_score(y[te], p[te], average="macro", zero_division=0)})
            preds[m].append(p)
    oof_store[size] = (ids, y, preds)
    # per-scenario OS-level result of each method, averaged over CV repeats
    tabs = {}
    for m in MODELS:
        reps = [method_table(ids, {m: p})[m] for p in preds[m]]
        tabs[m] = sum(reps) / len(reps)
    fixed = method_table(ids, {f"Fixed {b}": np.array([b] * len(ids)) for b in C.FIXED_BASELINES})
    tabs.update(fixed)
    tabs["Oracle(label set)"] = method_table(ids, {"o": d["best_policy"].values})["o"]
    tabs["Oracle(all 7, ref.)"] = method_table(ids, {"o": M["cost"].loc[ids].idxmin(axis=1).values})["o"]
    box_data[size] = {k: v["regret_pct"].values for k, v in tabs.items()}
    for m, t in tabs.items():
        lo, hi = boot_ci(t["regret_pct"].values)
        summary.append({"size": size, "method": m, "mean_avg_wait": t.avg_wait.mean(), "mean_avg_resp": t.avg_resp.mean(),
                        "mean_p95_wait": t.p95_wait.mean(), "mean_ctx_switches": t.ctx_switches.mean(),
                        "mean_regret_pct": t.regret_pct.mean(), "regret_ci_lo": lo, "regret_ci_hi": hi})
    # statistical tests: ML vs the best fixed baseline (paired, per-scenario regret)
    for bf in fixed:
        for m in MODELS:
            a, b = tabs[m]["regret_pct"].values, tabs[bf]["regret_pct"].values
            w = stats.wilcoxon(a, b) if np.any(a != b) else None
            stat_rows.append({"size": size, "ml_model": m, "vs": bf, "mean_regret_ml": a.mean(), "mean_regret_fixed": b.mean(),
                              "wilcoxon_p": None if w is None else float(w.pvalue)})

pd.DataFrame(accrows).to_csv(f"{RES}/cv_fold_scores.csv", index=False)
acc = pd.DataFrame(accrows).groupby(["size", "model"])[["accuracy", "macro_f1"]].agg(["mean", "std"]).round(4)
acc.to_csv(f"{RES}/classification_summary.csv")
S = pd.DataFrame(summary); S.to_csv(f"{RES}/oslevel_summary.csv", index=False)
pd.DataFrame(stat_rows).to_csv(f"{RES}/statistical_tests.csv", index=False)

# ---- cross-size generalisation: train on two sizes, test on the held-out size
cross = []
for held in C.SIZES:
    tr, te = feats[feats["size"] != held], feats[feats["size"] == held]
    for m in MODELS:
        clf = MODELS[m](C.SEED).fit(tr[FEAT_COLS].values, tr["best_policy"].values)
        p = clf.predict(te[FEAT_COLS].values)
        t = method_table(te.index.values, {m: p})[m]
        cross.append({"held_out_size": held, "model": m, "accuracy": accuracy_score(te["best_policy"], p),
                      "macro_f1": f1_score(te["best_policy"], p, average="macro", zero_division=0), "mean_regret_pct": t.regret_pct.mean()})
pd.DataFrame(cross).to_csv(f"{RES}/cross_size_generalisation.csv", index=False)

# ---- Task B: classify workload TYPE
tb, Xa, ya = [], feats[FEAT_COLS].values, feats["wtype"].values
type_oof = {}
for m in MODELS:
    p = np.empty(len(ya), dtype=object)
    for k, (tr, te) in enumerate(KFold(C.CV_FOLDS, shuffle=True, random_state=C.SEED).split(Xa)):
        p[te] = MODELS[m](C.SEED).fit(Xa[tr], ya[tr]).predict(Xa[te])
        tb.append({"model": m, "fold": k, "accuracy": accuracy_score(ya[te], p[te])})
    type_oof[m] = p
pd.DataFrame(tb).groupby("model")["accuracy"].agg(["mean", "std"]).round(4).to_csv(f"{RES}/workload_type_accuracy.csv")

# ---- feature importance (RF on everything)
rf = MODELS["RandomForest"](C.SEED).fit(feats[FEAT_COLS].values, feats["best_policy"].values)
imp = pd.Series(rf.feature_importances_, index=FEAT_COLS).sort_values()
imp.sort_values(ascending=False).to_csv(f"{RES}/rf_feature_importance.csv", header=["importance"])

# ================= plots =================
plt.rcParams.update({"font.size": 9, "figure.dpi": 130})
# 1 label distribution
ld = feats.groupby(["size", "best_policy"]).size().unstack(fill_value=0)[C.LABEL_POLICIES]
ld.plot(kind="bar", stacked=True, figsize=(5, 3.2)); plt.ylabel("scenarios"); plt.xlabel("workload size (processes)")
plt.title("Best-policy label distribution"); plt.tight_layout(); plt.savefig(f"{FIG}/fig1_label_distribution.png"); plt.close()
# 2 accuracy
a = pd.DataFrame(accrows).groupby(["size", "model"])["accuracy"].agg(["mean", "std"]).reset_index()
fig, ax = plt.subplots(figsize=(5.5, 3.2)); w = 0.25
for i, m in enumerate(MODELS):
    s = a[a.model == m]; ax.bar(np.arange(len(C.SIZES)) + i * w, s["mean"], w, yerr=s["std"], capsize=2, label=m)
ax.set_xticks(np.arange(len(C.SIZES)) + w); ax.set_xticklabels(C.SIZES); ax.set_ylim(0, 1); ax.set_ylabel("CV accuracy")
ax.set_xlabel("workload size"); ax.legend(fontsize=7); ax.set_title("Policy-prediction accuracy (5-fold x3)")
plt.tight_layout(); plt.savefig(f"{FIG}/fig2_accuracy.png"); plt.close()
# 3 regret bars
order = list(MODELS) + [f"Fixed {b}" for b in C.FIXED_BASELINES]
fig, ax = plt.subplots(figsize=(6.5, 3.3)); w = 0.8 / len(order)
for i, m in enumerate(order):
    s = S[S.method == m].set_index("size").loc[C.SIZES]
    ax.bar(np.arange(len(C.SIZES)) + i * w, s["mean_regret_pct"], w, label=m)
ax.set_xticks(np.arange(len(C.SIZES)) + 0.4 - w / 2); ax.set_xticklabels(C.SIZES); ax.set_ylabel("mean regret vs oracle (%)")
ax.set_xlabel("workload size"); ax.legend(fontsize=6, ncol=2); ax.set_title("OS-level cost: ML selector vs fixed policies (lower = better)")
plt.tight_layout(); plt.savefig(f"{FIG}/fig3_regret.png"); plt.close()
# 4 boxplot regret
fig, axs = plt.subplots(1, len(C.SIZES), figsize=(8, 3.2), sharey=True)
for ax, size in zip(axs, C.SIZES):
    ks = ["RandomForest", "DecisionTree", "Fixed FCFS", "Fixed RR_q10"]
    ax.boxplot([box_data[size][k] for k in ks], showfliers=False)
    ax.set_xticklabels(["RF", "DT", "FCFS", "RR10"]); ax.set_title(f"N={size}")
axs[0].set_ylabel("per-scenario regret (%)"); plt.tight_layout(); plt.savefig(f"{FIG}/fig4_regret_box.png"); plt.close()
# 5 confusion matrix (RF, pooled sizes, repeat 0)
yt = np.concatenate([oof_store[s][1] for s in C.SIZES]); yp = np.concatenate([oof_store[s][2]["RandomForest"][0] for s in C.SIZES])
cm = confusion_matrix(yt, yp, labels=C.LABEL_POLICIES)
fig, ax = plt.subplots(figsize=(4.2, 3.6)); ax.imshow(cm, cmap="Blues")
ax.set_xticks(range(len(C.LABEL_POLICIES))); ax.set_yticks(range(len(C.LABEL_POLICIES)))
ax.set_xticklabels(C.LABEL_POLICIES, rotation=45); ax.set_yticklabels(C.LABEL_POLICIES)
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]): ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=8)
ax.set_xlabel("predicted"); ax.set_ylabel("true best"); ax.set_title("Random Forest confusion matrix"); plt.tight_layout()
plt.savefig(f"{FIG}/fig5_confusion_rf.png"); plt.close()
# 6 feature importance
imp.tail(10).plot(kind="barh", figsize=(5, 3.2)); plt.xlabel("RF importance"); plt.title("Top features"); plt.tight_layout()
plt.savefig(f"{FIG}/fig6_feature_importance.png"); plt.close()

pd.set_option("display.width", 200); pd.set_option("display.max_columns", 20)
print("\n== Classification (policy/quantum prediction) =="); print(acc)
print("\n== OS-level summary (regret % vs oracle on label set) ==")
print(S.pivot(index="method", columns="size", values="mean_regret_pct").round(2))
print("\n== Mean avg waiting time =="); print(S.pivot(index="method", columns="size", values="mean_avg_wait").round(0))
print("\n== Mean avg response time =="); print(S.pivot(index="method", columns="size", values="mean_avg_resp").round(1))
print("\n== Wilcoxon (ML vs best fixed) =="); print(pd.DataFrame(stat_rows).to_string(float_format=lambda v: f"{v:.3g}"))
print("\n== Cross-size generalisation =="); print(pd.DataFrame(cross).round(3))
print("\n== Workload-type accuracy =="); print(pd.read_csv(f"{RES}/workload_type_accuracy.csv"))
print("\n== Top features =="); print(imp.tail(6)[::-1].round(3))

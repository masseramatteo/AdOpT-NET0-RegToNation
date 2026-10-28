"""
Test 2 on the v5 campaign: are the DT_uncert conclusions robust to the scenario design?

The v5 design is paired (8 runs per economy) with equal archetype shares (~25% each). A flat
LHS over the same box would give one run per economy and archetype shares of about
A 30 / B 31 / C 1 / D 37 % (scenario-design defence, 2026-09-20). No flat-LHS campaign was run,
so it is imitated by resampling the existing runs:

    flat-LHS  : per economy keep ONE run; archetype drawn A .30 / B .31 / C .01 / D .37,
                near/far 50/50                                          -> ~749 runs
    control   : same, archetype drawn 25/25/25/25 (v5 shares)           -> ~749 runs

The control has the same n and the same one-run-per-economy structure, so
    control  vs DT_uncert -> effect of n alone
    flat-LHS vs control   -> effect of the archetype shares (the question)

Labels are the DT_uncert cluster labels recomputed on all runs (same KMeans, RS=42) and held
fixed; trees use the DT_uncert (max_leaf_nodes, min_samples_leaf) per case, no re-tuning.
One run per economy -> plain stratified 5-fold CV (no grouping needed). The same economy->run
draw is used for every case within a draw (common random numbers).

Decision criteria fixed before running (network trees, topo/combo):
    flag_dss   flat-LHS median d_SS importance outside the control 90% interval
    flag_kappa |flat-LHS median kappa - control median kappa| > 0.05
    flag_pass  PASS (kappa >= 0.60, baseline <= 0.80) differs between flat-LHS and control
Any flag on a network tree -> consider a real flat-LHS campaign.

Outputs (into <results>/DT_flatlhs):
    draws_long.csv, comparison.csv, kappa_dss_scatter.png, network_cases_forest.png

Run:  python dt_v5_flatlhs_resample.py
"""
import os

# Limit library thread pools; parallelism is over cases (joblib).
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, balanced_accuracy_score, cohen_kappa_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler

import config as C
import dt_v5 as V

RS = V.RS
NDRAW = 200
OUT = os.path.join(C.RESULTS_FOLDER, "DT_flatlhs")
REF = os.path.join(C.RESULTS_FOLDER, "DT_uncert", "comparison_all.csv")
DRIVERS = [d for d in V.DRIVERS if d != "k_small"]   # as in dt_v5_uncertain.py
DIST = V.DIST
SHARES = {"flatlhs": {"A": 0.30, "B": 0.31, "C": 0.01, "D": 0.37},
          "control": {"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25}}
NETWORK = ("topo", "combo")
NAVY, TEAL, AMBER, RED = V.UU_NAVY, V.UU_TEAL, V.UU_AMBER, V.UU_RED


def labels(f, feats, k):
    """DT_uncert labels: same scaler, KMeans and ordering as dt_v5_uncertain.main()."""
    Xc = StandardScaler().fit_transform(f[feats].fillna(0))
    y = KMeans(k, random_state=RS, n_init=10).fit(Xc).labels_
    order = np.argsort([f[feats[0]].values[y == c].mean() for c in range(k)])
    remap = {o: i for i, o in enumerate(order)}
    return np.array([remap[c] for c in y])


def draws(f, scheme, n, seed):
    """n index arrays, one run per economy, archetype drawn with SHARES[scheme]."""
    rng = np.random.default_rng(seed)
    p = pd.Series(SHARES[scheme])
    p = p / p.sum()
    slot_idx = f.reset_index().set_index(["economy", "slot"])["index"]
    econ = np.sort(f.economy.unique())
    out = []
    for _ in range(n):
        pick = []
        for e in econ:
            avail = slot_idx.loc[e]
            while True:
                a = rng.choice(p.index, p=p.values)
                sides = [s for s in (f"{a}_near", f"{a}_far") if s in avail.index]
                if sides:
                    pick.append(avail[sides[rng.integers(len(sides))]])
                    break
        out.append(np.array(pick))
    return out


def fit_case(case, y, X, idx_sets):
    """kappa/acc/bal_acc (5-fold CV) and DT importance on each resampled index set."""
    rows = []
    for scheme, sets in idx_sets.items():
        for i, ix in enumerate(sets):
            Xs, ys = X.iloc[ix], y[ix]
            # a class can vanish or be tiny in a subsample; CV needs >= 5 per class
            vc = pd.Series(ys).value_counts()
            keep = np.isin(ys, vc[vc >= 5].index)
            Xs, ys = Xs[keep], ys[keep]
            dt = V.mk_tree(case["leaves"], case["min_leaf"])
            p = cross_val_predict(dt, Xs, ys, cv=StratifiedKFold(5, shuffle=True, random_state=RS))
            imp = pd.Series(dt.fit(Xs, ys).feature_importances_ * 100, index=DRIVERS)
            rows.append(dict(group=case["group"], definition=case["definition"], k=case["k"],
                             scheme=scheme, draw=i, n=len(ys),
                             n_classes=len(np.unique(ys)),
                             baseline=pd.Series(ys).value_counts(normalize=True).max(),
                             accuracy=accuracy_score(ys, p),
                             bal_accuracy=balanced_accuracy_score(ys, p),
                             kappa=cohen_kappa_score(ys, p),
                             dss_pct=imp["d_SS"], dist_pct=imp[DIST].sum()))
    return rows


def reference(case, y, X):
    """DT_uncert tree refit on all runs: d_SS importance (not stored in comparison_all.csv)."""
    imp = pd.Series(V.mk_tree(case["leaves"], case["min_leaf"]).fit(X, y).feature_importances_ * 100,
                    index=DRIVERS)
    return imp["d_SS"], imp[DIST].sum()


def summarise(L, ref):
    q = lambda s, a: s.quantile(a)
    agg = L.groupby(["group", "definition", "k", "scheme"]).agg(
        kappa_med=("kappa", "median"), kappa_lo=("kappa", lambda s: q(s, .05)),
        kappa_hi=("kappa", lambda s: q(s, .95)),
        acc_med=("accuracy", "median"), bal_med=("bal_accuracy", "median"),
        base_med=("baseline", "median"),
        dss_med=("dss_pct", "median"), dss_lo=("dss_pct", lambda s: q(s, .05)),
        dss_hi=("dss_pct", lambda s: q(s, .95)),
        dist_med=("dist_pct", "median"), n_med=("n", "median")).reset_index()
    w = agg.pivot_table(index=["group", "definition", "k"], columns="scheme")
    w.columns = [f"{s}_{m}" for m, s in w.columns]
    w = ref.merge(w.reset_index(), on=["group", "definition", "k"])
    w["d_kappa_flat_vs_control"] = w.flatlhs_kappa_med - w.control_kappa_med
    w["d_kappa_control_vs_ref"] = w.control_kappa_med - w.kappa
    w["flag_dss"] = (w.flatlhs_dss_med < w.control_dss_lo) | (w.flatlhs_dss_med > w.control_dss_hi)
    w["flag_kappa"] = w.d_kappa_flat_vs_control.abs() > 0.05
    pas = lambda s: (w[f"{s}_kappa_med"] >= 0.60) & (w[f"{s}_base_med"] <= 0.80)
    w["flag_pass"] = pas("flatlhs") != pas("control")
    w["network"] = w.group.isin(NETWORK)
    return w


def fig_scatter(w):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    gc = {"sysmix": "0.6", "decentr": AMBER, "topo": NAVY, "combo": TEAL}
    for ax, (m, lab) in zip(axes, [("kappa", "kappa (CV)"), ("dss", "d_SS importance (%)")]):
        for g, s in w.groupby("group"):
            ax.errorbar(s[f"control_{m}_med"], s[f"flatlhs_{m}_med"],
                        xerr=[s[f"control_{m}_med"] - s[f"control_{m}_lo"],
                              s[f"control_{m}_hi"] - s[f"control_{m}_med"]],
                        fmt="o", ms=4, color=gc[g], ecolor=gc[g], elinewidth=0.6, alpha=0.8,
                        label=g)
        lo = min(ax.get_xlim()[0], ax.get_ylim()[0])
        hi = max(ax.get_xlim()[1], ax.get_ylim()[1])
        ax.plot([lo, hi], [lo, hi], "--", color="grey", lw=0.8)
        if m == "kappa":
            ax.fill_between([lo, hi], [lo - 0.05, hi - 0.05], [lo + 0.05, hi + 0.05],
                            color="0.92", zorder=0)
        ax.set_xlabel(f"Control (v5 shares, 1 run/economy): median {lab}")
        ax.set_ylabel(f"Flat-LHS shares: median {lab}")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(fontsize=8, frameon=False)
    fig.suptitle(f"Flat-LHS archetype shares vs control, all {len(w)} DT_uncert cases "
                 f"({NDRAW} draws each)", color=NAVY, fontsize=12, x=0.02, ha="left")
    fig.text(0.02, -0.03, "Error bars = control 90% interval over draws; grey band = +/-0.05 kappa. "
             "Points on the diagonal: archetype shares do not change the result.",
             fontsize=7.5, color="grey")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "kappa_dss_scatter.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_forest(w):
    s = w[w.network].sort_values("dss_ref", ascending=False).head(15).iloc[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
    ylab = [f"{r.group}/{r.definition} k{r.k}" for r in s.itertuples()]
    for ax, (m, ref, lab) in zip(axes, [("dss", "dss_ref", "d_SS importance (%)"),
                                        ("kappa", "kappa", "kappa (CV)")]):
        yy = np.arange(len(s))
        ax.plot(s[ref], yy, "D", color=RED, ms=5, label="DT_uncert (all 5992 runs)")
        for off, sc, col in [(-0.15, "control", "0.55"), (0.15, "flatlhs", NAVY)]:
            ax.hlines(yy + off, s[f"{sc}_{m}_lo"], s[f"{sc}_{m}_hi"], color=col, lw=2)
            ax.plot(s[f"{sc}_{m}_med"], yy + off, "o", color=col, ms=5,
                    label="control (v5 shares)" if sc == "control" else "flat-LHS shares")
        if m == "kappa":
            ax.axvline(0.60, color="grey", ls="--", lw=0.8)
        ax.set_xlabel(lab)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_yticks(np.arange(len(s)))
    axes[0].set_yticklabels(ylab, fontsize=8)
    axes[0].legend(fontsize=8, frameon=False, loc="lower right")
    fig.suptitle("Network trees with the largest d_SS role: design robustness",
                 color=NAVY, fontsize=12, x=0.02, ha="left")
    fig.text(0.02, -0.02, f"Median and 90% interval over {NDRAW} draws of one run per economy "
             "(~749 runs). Dashed line: PASS threshold kappa 0.60.", fontsize=7.5, color="grey")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "network_cases_forest.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    f = V.load(C.EXTRACTED_RESULTS).reset_index(drop=True)
    X = f[DRIVERS].fillna(0.0)
    ref = pd.read_csv(REF)[["group", "definition", "k", "n", "baseline", "kappa", "leaves",
                            "min_leaf", "dist_pct", "PASS"]]
    feats = {(g, n): ft for g, defs in V.GROUPS.items() for n, ft, _ in defs}
    print(f"{len(f)} runs, {f.economy.nunique()} economies, {len(ref)} cases")

    idx_sets = {"flatlhs": draws(f, "flatlhs", NDRAW, RS),
                "control": draws(f, "control", NDRAW, RS + 1)}
    for s, sets in idx_sets.items():
        arch = pd.Series(np.concatenate([f.slot.values[ix] for ix in sets])).str[0]
        print(f"  {s}: n/draw {len(sets[0])}, archetype shares "
              + " ".join(f"{a} {v:.2f}" for a, v in arch.value_counts(normalize=True).sort_index().items()))

    cases, Ys = [], []
    for case in ref.to_dict("records"):
        y = labels(f, feats[(case["group"], case["definition"])], case["k"])
        base = pd.Series(y).value_counts(normalize=True).max()
        if abs(base - case["baseline"]) > 0.002:
            print(f"  WARNING label mismatch {case['group']}/{case['definition']} k{case['k']}: "
                  f"baseline {base:.3f} vs DT_uncert {case['baseline']:.3f}")
        case["dss_ref"], case["dist_ref"] = reference(case, y, X)
        cases.append(case)
        Ys.append(y)
    print(f"  labels recomputed; DT_uncert dist_pct reproduced within 0.5 pt in "
          f"{sum(abs(c['dist_ref'] - c['dist_pct']) < 0.5 for c in cases)}/{len(cases)} cases")

    res = Parallel(n_jobs=-1, verbose=5)(delayed(fit_case)(c, y, X, idx_sets)
                                         for c, y in zip(cases, Ys))
    L = pd.DataFrame([r for rr in res for r in rr])
    L.round(4).to_csv(os.path.join(OUT, "draws_long.csv"), index=False)

    w = summarise(L, pd.DataFrame(cases).drop(columns=["n"]))
    w.round(4).to_csv(os.path.join(OUT, "comparison.csv"), index=False)
    fig_scatter(w)
    fig_forest(w)

    net = w[w.network]
    print(f"\nNetwork trees ({len(net)}): flag_dss {int(net.flag_dss.sum())}, "
          f"flag_kappa {int(net.flag_kappa.sum())}, flag_pass {int(net.flag_pass.sum())}")
    print(f"All trees   ({len(w)}): flag_dss {int(w.flag_dss.sum())}, "
          f"flag_kappa {int(w.flag_kappa.sum())}, flag_pass {int(w.flag_pass.sum())}")
    print(f"Median kappa drop from n alone (control - DT_uncert): "
          f"{w.d_kappa_control_vs_ref.median():+.3f}")
    cols = ["group", "definition", "k", "kappa", "control_kappa_med", "flatlhs_kappa_med",
            "dss_ref", "control_dss_med", "flatlhs_dss_med", "flag_dss", "flag_kappa", "flag_pass"]
    print(net.sort_values("dss_ref", ascending=False)[cols].head(15).round(3).to_string(index=False))
    print(f"\nsaved to {OUT}")


if __name__ == "__main__":
    main()

"""
Out-of-sample validation of the v5 DT_uncert results on the flat-LHS campaign (2026-11).

The flat-LHS campaign (1500 runs, 1 per economy, new economies and new geometry over the
v5 box, archetype drawn among those that can build the triple) was never seen by the v5
analysis. Protocol and criteria fixed before looking at the outcomes:

A. Tree transfer, all DT_uncert cases (group / definition / k)
   labels  v5 StandardScaler + KMeans (RS, as DT_uncert) fitted on v5; flat-LHS runs get
           the label of the nearest v5 centroid -> identical class meaning in both sets
   A1      tree fitted on all v5 runs (DT_uncert leaves / min_leaf) predicts flat-LHS
           -> kappa_transfer vs v5 grouped-CV kappa
           flag_transfer : kappa_transfer < kappa_v5 - 0.05
           flag_pass     : PASS (kappa >= 0.60, baseline <= 0.80) differs v5 vs transfer
   A2      same tree settings refitted on flat-LHS, 5-fold stratified CV (1 run/economy)
           -> kappa_flatcv; importance of d_SS and of the three distances vs v5 refit.
           n is 1495 vs 5992: test 2 measured -0.07 kappa for n 749, so a drop here is
           partly n and is reported, not flagged.
B. Findings
   B1      "distances shape the network layout, not the supply mix": median distance
           importance of network groups (topo, combo) > supply groups (sysmix, decentr)
           in the flat-LHS refits, as in v5.
   B2      "d_SS drives the small-small link": linear probability model
               SS_pipe ~ d_SS + d_SL_mean + d_LL (per +100 km) + standardised economics
           on flat-LHS (bootstrap over runs) and on v5 (bootstrap over economies),
           pooled and for A/B vs D. Replicated if the flat-LHS d_SS slope is < 0 with a
           95% CI excluding 0 and overlapping the v5 CI.
           Reference, v5 within-economy (test 1): A/B -0.060, D -0.042 per +100 km.
   B3      "archetype adds nothing beyond distances": network cases, flat-LHS CV kappa
           with the archetype one-hot added to the drivers; replicated if the median
           gain is <= 0.02.

Outputs (into <flat-LHS results>/DT_validation):
    transfer_cases.csv, findings_B1_distance_share.csv, findings_B2_dss_slope.csv,
    findings_B3_archetype.csv, summary.txt, transfer_kappa.png, dss_slope.png

Run:  python dt_v5_flatlhs_validate.py   (after postprocessing/extract_flatlhs.py)
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
FLAT_FOLDER = (
    r"\\soliscom.uu.nl\geo\SD\Energy and Resources\GazzaniGroup\Matteo M"
    r"\AdOpT-NET0-RegToNation\four_node_configuration\results"
    r"\flatlhs_validation_20261028_171519"
)
OUT = os.path.join(FLAT_FOLDER, "DT_validation")
REF = os.path.join(C.RESULTS_FOLDER, "DT_uncert", "comparison_all.csv")
DRIVERS = [d for d in V.DRIVERS if d != "k_small"]   # as in dt_v5_uncertain.py
DIST = V.DIST
NETWORK = ("topo", "combo")
ECON = [d for d in DRIVERS if d not in DIST]
NBOOT = 1000
NAVY, TEAL, AMBER, RED = V.UU_NAVY, V.UU_TEAL, V.UU_AMBER, V.UU_RED
GCOL = {"sysmix": "0.6", "decentr": AMBER, "topo": NAVY, "combo": TEAL}


# ------------------------------------------------------------------ A: trees --
def v5_labels(f5, ff, feats, k):
    """DT_uncert labels on v5 and nearest-v5-centroid labels on flat-LHS."""
    sc = StandardScaler().fit(f5[feats].fillna(0))
    km = KMeans(k, random_state=RS, n_init=10).fit(sc.transform(f5[feats].fillna(0)))
    y5 = km.labels_
    order = np.argsort([f5[feats[0]].values[y5 == c].mean() for c in range(k)])
    remap = np.empty(k, int)
    remap[order] = np.arange(k)
    yf = km.predict(sc.transform(ff[feats].fillna(0)))
    return remap[y5], remap[yf]


def importance(case, X, y):
    imp = pd.Series(V.mk_tree(case["leaves"], case["min_leaf"]).fit(X, y).feature_importances_
                    * 100, index=X.columns)
    return imp


def flat_cv(case, X, y):
    vc = pd.Series(y).value_counts()
    keep = np.isin(y, vc[vc >= 5].index)       # CV needs >= 5 runs per class
    p = cross_val_predict(V.mk_tree(case["leaves"], case["min_leaf"]), X[keep], y[keep],
                          cv=StratifiedKFold(5, shuffle=True, random_state=RS))
    return y[keep], p


def run_case(case, X5, Xf, Xf_arch, y5, yf):
    tree = V.mk_tree(case["leaves"], case["min_leaf"]).fit(X5, y5)
    pt = tree.predict(Xf)
    imp5 = pd.Series(tree.feature_importances_ * 100, index=DRIVERS)
    yc, pc = flat_cv(case, Xf, yf)
    impf = importance(case, Xf, yf)
    row = dict(group=case["group"], definition=case["definition"], k=case["k"],
               kappa_v5=case["kappa"], acc_v5=case["accuracy"], bal_v5=case["bal_accuracy"],
               base_v5=case["baseline"], PASS_v5=case["PASS"],
               n_flat=len(yf), n_classes_flat=len(np.unique(yf)),
               base_flat=pd.Series(yf).value_counts(normalize=True).max(),
               kappa_transfer=cohen_kappa_score(yf, pt),
               acc_transfer=accuracy_score(yf, pt),
               bal_transfer=balanced_accuracy_score(yf, pt),
               kappa_flatcv=cohen_kappa_score(yc, pc), acc_flatcv=accuracy_score(yc, pc),
               bal_flatcv=balanced_accuracy_score(yc, pc),
               dss_v5=imp5["d_SS"], dist_v5=imp5[DIST].sum(), top_v5=imp5.idxmax(),
               dss_flat=impf["d_SS"], dist_flat=impf[DIST].sum(), top_flat=impf.idxmax())
    # class shares: v5 vs flat-LHS (labels from the same centroids)
    s5 = pd.Series(y5).value_counts(normalize=True)
    sf = pd.Series(yf).value_counts(normalize=True)
    row["share_shift_max"] = float((s5 - sf.reindex(s5.index, fill_value=0)).abs().max())
    if case["group"] in NETWORK:
        yca, pca = flat_cv(case, Xf_arch, yf)
        row["kappa_flatcv_arch"] = cohen_kappa_score(yca, pca)
    return row


# --------------------------------------------------------------- B2: d_SS slope --
def design(f, z_mu, z_sd):
    X = np.column_stack([np.ones(len(f)), f.d_SS / 100, f.d_SL_mean / 100, f.d_LL / 100,
                         (f[ECON].fillna(0) - z_mu) / z_sd])
    return X, f.SS_pipe.to_numpy(float)


def slope(f, z_mu, z_sd):
    X, y = design(f, z_mu, z_sd)
    return np.linalg.lstsq(X, y, rcond=None)[0][1]


def boot_slope(f, z_mu, z_sd, cluster, rng):
    est = slope(f, z_mu, z_sd)
    if cluster:
        idx = f.groupby("economy").indices
        econ = np.array(list(idx))
        draws = []
        for _ in range(NBOOT):
            pick = np.concatenate([idx[e] for e in rng.choice(econ, len(econ))])
            draws.append(slope(f.iloc[pick], z_mu, z_sd))
    else:
        draws = [slope(f.iloc[rng.integers(0, len(f), len(f))], z_mu, z_sd) for _ in range(NBOOT)]
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return est, lo, hi


def b2(f5, ff):
    rng = np.random.default_rng(RS)
    z_mu, z_sd = f5[ECON].mean(), f5[ECON].std()         # same scaling on both sets
    rows = []
    sub = {"pooled": lambda f: f.archetype.isin(list("ABCD")),
           "A/B": lambda f: f.archetype.isin(["A", "B"]),
           "D": lambda f: f.archetype == "D"}
    for name, m in sub.items():
        for camp, f, cl in (("v5", f5, True), ("flatlhs", ff, False)):
            g = f[m(f)]
            est, lo, hi = boot_slope(g, z_mu, z_sd, cl, rng)
            rows.append(dict(subset=name, campaign=camp, n=len(g), ss_rate=g.SS_pipe.mean(),
                             slope=est, lo=lo, hi=hi))
    R = pd.DataFrame(rows)
    w = R.pivot(index="subset", columns="campaign", values=["slope", "lo", "hi"])
    w.columns = [f"{m}_{c}" for m, c in w.columns]
    w["replicated"] = (w.hi_flatlhs < 0) & (w.lo_flatlhs <= w.hi_v5) & (w.hi_flatlhs >= w.lo_v5)
    return R, w.reset_index()


# ------------------------------------------------------------------- figures --
def fig_transfer(T):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
    for ax, (col, lab) in zip(axes, [("kappa_transfer", "v5 tree applied to flat-LHS"),
                                     ("kappa_flatcv", "refitted on flat-LHS (5-fold CV)")]):
        for g, s in T.groupby("group"):
            ax.plot(s.kappa_v5, s[col], "o", ms=4, color=GCOL[g], alpha=0.8, label=g)
        ax.plot([0, 1], [0, 1], "--", color="grey", lw=0.8)
        ax.fill_between([0, 1], [-0.05, 0.95], [0.05, 1.05], color="0.92", zorder=0)
        ax.axvline(0.60, color="grey", ls=":", lw=0.8)
        ax.axhline(0.60, color="grey", ls=":", lw=0.8)
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.05, 1)
        ax.set_xlabel("v5 grouped-CV kappa (DT_uncert)")
        ax.set_title(lab, fontsize=10, color=NAVY)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("kappa on flat-LHS")
    axes[0].legend(fontsize=8, frameon=False)
    fig.suptitle(f"Out-of-sample check of the v5 trees on the flat-LHS campaign, {len(T)} cases",
                 color=NAVY, fontsize=12, x=0.02, ha="left")
    fig.text(0.02, -0.03, "Grey band = +/-0.05 kappa; dotted = PASS threshold 0.60. Labels from "
             "the v5 cluster centroids in both sets.", fontsize=7.5, color="grey")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "transfer_kappa.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_slope(R):
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    subs = ["pooled", "A/B", "D"]
    for j, (camp, col) in enumerate((("v5", "0.55"), ("flatlhs", NAVY))):
        s = R[R.campaign == camp].set_index("subset").loc[subs]
        yy = np.arange(len(subs)) + (j - 0.5) * 0.25
        ax.hlines(yy, s.lo, s.hi, color=col, lw=2)
        ax.plot(s.slope, yy, "o", color=col, ms=6,
                label="v5 (cross-section, CI over economies)" if camp == "v5" else "flat-LHS")
    for a, v in (("A/B", -0.060), ("D", -0.042)):
        ax.plot(v, subs.index(a), "D", color=RED, ms=5,
                label="v5 within-economy (test 1)" if a == "A/B" else None)
    ax.axvline(0, color="grey", lw=0.8)
    ax.set_yticks(range(len(subs)))
    ax.set_yticklabels(subs)
    ax.invert_yaxis()
    ax.set_xlabel("Change in P(small-small link) per +100 km d_SS")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=7.5, frameon=False, loc="lower left")
    ax.set_title("d_SS effect on the small-small link: v5 vs flat-LHS", fontsize=10, color=NAVY)
    fig.text(0.02, -0.06, "Linear probability model with d_SL, d_LL and standardised economics as "
             f"controls; 95% CI, {NBOOT} bootstrap draws.", fontsize=7, color="grey")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "dss_slope.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------- main --
def main():
    os.makedirs(OUT, exist_ok=True)
    f5 = V.load(C.EXTRACTED_RESULTS).reset_index(drop=True)
    ff = V.load(os.path.join(FLAT_FOLDER, "extracted_results.xlsx")).reset_index(drop=True)
    X5, Xf = f5[DRIVERS].fillna(0.0), ff[DRIVERS].fillna(0.0)
    arch = pd.get_dummies(ff.archetype, prefix="arch").astype(float)
    Xf_arch = pd.concat([Xf, arch], axis=1)
    ref = pd.read_csv(REF)
    feats = {(g, n): ft for g, defs in V.GROUPS.items() for n, ft, _ in defs}
    print(f"v5 {len(f5)} runs | flat-LHS {len(ff)} runs, archetypes "
          f"{ff.archetype.value_counts().to_dict()} | {len(ref)} cases")

    cases, labs = [], []
    for case in ref.to_dict("records"):
        y5, yf = v5_labels(f5, ff, feats[(case["group"], case["definition"])], case["k"])
        base = pd.Series(y5).value_counts(normalize=True).max()
        if abs(base - case["baseline"]) > 0.002:
            print(f"  WARNING label mismatch {case['group']}/{case['definition']} k{case['k']}: "
                  f"baseline {base:.3f} vs DT_uncert {case['baseline']:.3f}")
        cases.append(case)
        labs.append((y5, yf))
    rows = Parallel(n_jobs=-1, verbose=5)(delayed(run_case)(c, X5, Xf, Xf_arch, y5, yf)
                                          for c, (y5, yf) in zip(cases, labs))
    T = pd.DataFrame(rows)
    T["d_kappa_transfer"] = T.kappa_transfer - T.kappa_v5
    T["d_kappa_flatcv"] = T.kappa_flatcv - T.kappa_v5
    T["flag_transfer"] = T.d_kappa_transfer < -0.05
    T["PASS_transfer"] = ((T.kappa_transfer >= 0.60) & (T.base_flat <= 0.80)).astype(int)
    T["flag_pass"] = T.PASS_transfer != T.PASS_v5
    T["network"] = T.group.isin(NETWORK)
    T.round(4).to_csv(os.path.join(OUT, "transfer_cases.csv"), index=False)

    B1 = T.groupby("group")[["dist_v5", "dist_flat", "dss_v5", "dss_flat"]].median().round(1)
    B1.to_csv(os.path.join(OUT, "findings_B1_distance_share.csv"))
    b1_ok = (B1.loc[list(NETWORK), "dist_flat"].min()
             > B1.drop(index=list(NETWORK)).dist_flat.max())

    R, W = b2(f5, ff)
    R.round(4).to_csv(os.path.join(OUT, "findings_B2_dss_slope.csv"), index=False)

    net = T[T.network].copy()
    net["arch_gain"] = net.kappa_flatcv_arch - net.kappa_flatcv
    net[["group", "definition", "k", "kappa_flatcv", "kappa_flatcv_arch", "arch_gain"]].round(4) \
        .to_csv(os.path.join(OUT, "findings_B3_archetype.csv"), index=False)

    fig_transfer(T)
    fig_slope(R)

    P = T[T.PASS_v5 == 1]
    lines = [
        f"Flat-LHS validation of v5 DT_uncert | v5 {len(f5)} runs, flat-LHS {len(ff)} runs",
        "",
        f"A1 transfer (v5 tree -> flat-LHS), {len(T)} cases:",
        f"   d kappa (transfer - v5 CV): median {T.d_kappa_transfer.median():+.3f}, "
        f"network {net.d_kappa_transfer.median():+.3f}",
        f"   flag_transfer (< -0.05): {int(T.flag_transfer.sum())}/{len(T)} "
        f"(network {int(net.flag_transfer.sum())}/{len(net)})",
        f"   flag_pass: {int(T.flag_pass.sum())}; v5 PASS cases {len(P)}, still PASS on flat-LHS "
        f"{int(P.PASS_transfer.sum())}",
        f"A2 refit on flat-LHS (n {len(ff)}): d kappa median {T.d_kappa_flatcv.median():+.3f}",
        "",
        "v5 PASS cases (kappa | acc | bal acc | baseline):",
    ] + [f"   {r.group}/{r.definition} k{r.k}: v5 {r.kappa_v5:.3f} | {r.acc_v5:.3f} | "
         f"{r.bal_v5:.3f} | {r.base_v5:.3f}  ->  transfer {r.kappa_transfer:.3f} | "
         f"{r.acc_transfer:.3f} | {r.bal_transfer:.3f} | {r.base_flat:.3f}  ->  flat CV "
         f"{r.kappa_flatcv:.3f} | {r.acc_flatcv:.3f} | {r.bal_flatcv:.3f}"
         for r in P.itertuples()] + [
        "",
        f"B1 distance share of DT importance (median %), network > supply groups on flat-LHS: "
        f"{'REPLICATED' if b1_ok else 'NOT replicated'}",
        B1.to_string(),
        "",
        "B2 d_SS slope on SS_pipe per +100 km (95% CI):",
        W.round(4).to_string(index=False),
        "",
        f"B3 archetype one-hot added (network cases, flat-LHS CV): median kappa gain "
        f"{net.arch_gain.median():+.3f}, max {net.arch_gain.max():+.3f} -> "
        f"{'REPLICATED' if net.arch_gain.median() <= 0.02 else 'NOT replicated'}",
    ]
    report = "\n".join(lines)
    with open(os.path.join(OUT, "summary.txt"), "w", encoding="utf-8") as fh:
        fh.write(report + "\n")
    print("\n" + report + f"\n\nsaved to {OUT}")


if __name__ == "__main__":
    main()

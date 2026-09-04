"""
Cluster on the LEVEL OF DECENTRALISATION only (1-D), k chosen by elbow + silhouette,
then predict the decentralisation cluster from the sampled inputs with a small tree.

Two definitions of decentralisation (both tried):
  A) small_prod / total_demand   — share of the whole system's H2 demand made locally
  B) small_prod / small_demand   — small clusters' self-sufficiency

For each: elbow + silhouette over k=2..8, pick k (max silhouette), k-Means, order the
clusters low->high, report their ranges, then a driver tree (inputs -> cluster) with
CV + 90/10 test + baseline + high-purity (low-gini) rules.

Outputs: DT_supplychain/decentralization_only/{A,B}/

Run:  python decentralization_clustering.py
"""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import cross_val_score, train_test_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C
from pure_leaves import leaf_rules
from robust_trees import robust_run_ids, ROBUST_SCENARIO

# write via the X: mapped drive (avoids UNC issues, per project convention)
OUT = r"X:\3000_simulations_24_may_2026\Snellius_3000_simulations\DT_supplychain\decentralization_only"
NODES = ["Large_cluster1", "Large_cluster2", "Small_cluster1", "Small_cluster2"]
SMALL = ["Small_cluster1", "Small_cluster2"]
KRANGE = range(2, 9)
PURITY = 0.70


def load_metrics():
    raw = pd.read_excel(C.EXTRACTED_RESULTS)
    fc = []
    for n in NODES:
        fc += [f"{n}_TOTAL_H2_production", f"{n}_hydrogen_import_sum",
               f"{n}_hydrogen_network_inflow_sum", f"{n}_hydrogen_network_outflow_sum"]
    raw[fc] = raw[fc].fillna(0.0)
    prod = {n: raw[f"{n}_TOTAL_H2_production"] for n in NODES}
    dem = {n: prod[n] + raw[f"{n}_hydrogen_import_sum"] + raw[f"{n}_hydrogen_network_inflow_sum"]
           - raw[f"{n}_hydrogen_network_outflow_sum"] for n in NODES}
    small_prod = sum(prod[n] for n in SMALL)
    small_dem = sum(dem[n] for n in SMALL)
    total_dem = sum(dem[n] for n in NODES)
    df = pd.DataFrame({"run": raw["run"]})
    df["defA"] = (small_prod / total_dem.replace(0, np.nan)).fillna(0)
    df["defB"] = (small_prod / small_dem.replace(0, np.nan)).clip(upper=1.0).fillna(0)  # self-suff, cap 1
    return df


def choose_k(x):
    xs = StandardScaler().fit_transform(x.reshape(-1, 1))
    inertias, sils = [], []
    for k in KRANGE:
        km = KMeans(k, random_state=C.RANDOM_STATE, n_init=10).fit(xs)
        inertias.append(km.inertia_)
        sils.append(silhouette_score(xs, km.labels_))
    best_k = list(KRANGE)[int(np.argmax(sils))]
    return xs, list(KRANGE), inertias, sils, best_k


def run_def(feat, dm, name, metric_col, tag):
    out = os.path.join(OUT, name, tag)
    os.makedirs(out, exist_ok=True)
    x = dm[metric_col].values
    xs, ks, inertias, sils, best_k = choose_k(x)

    # elbow + silhouette figure
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4))
    a1.plot(ks, inertias, "o-", color=C.UU_NAVY); a1.set_title("Elbow"); a1.set_xlabel("k"); a1.grid(alpha=.3)
    a2.plot(ks, sils, "o-", color=C.UU_TEAL); a2.set_title("Silhouette"); a2.set_xlabel("k"); a2.grid(alpha=.3)
    a2.axvline(best_k, color=C.UU_RED, ls="--")
    fig.suptitle(f"Decentralisation def {name}  ({metric_col})  -> k*={best_k}")
    fig.tight_layout(); fig.savefig(os.path.join(out, "elbow_silhouette.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)

    km = KMeans(best_k, random_state=C.RANDOM_STATE, n_init=10).fit(xs)
    lab = km.labels_
    # order clusters low->high by metric mean
    order = np.argsort([x[lab == c].mean() for c in range(best_k)])
    remap = {old: new for new, old in enumerate(order)}
    y = np.array([remap[c] for c in lab])
    cls_names = [f"D{i}" for i in range(best_k)]

    print(f"\n=== def {name} [{tag}] ({metric_col}) n={len(dm)}  k*={best_k} (silhouette {max(sils):.3f}) ===")
    print("  elbow inertias:", [f"{v:.0f}" for v in inertias])
    print("  cluster ranges (low->high):")
    for i in range(best_k):
        xi = x[y == i]
        print(f"    D{i}: n={len(xi):>4}  {metric_col} in [{xi.min():.3f}, {xi.max():.3f}]  mean {xi.mean():.3f}")

    # driver tree inputs -> decentralisation cluster
    df = feat.merge(dm[["run", metric_col]], on="run", how="left")
    inputs = [c for c in C.DRIVER_INPUTS if c in df.columns]
    X = df[inputs].fillna(0).reset_index(drop=True)
    base = pd.Series(y).value_counts(normalize=True).max()
    leaves = best_k + 3
    clf = DecisionTreeClassifier(max_leaf_nodes=leaves, min_samples_leaf=20, min_samples_split=20,
                                 class_weight="balanced", random_state=C.RANDOM_STATE)
    cv = cross_val_score(clf, X, y, cv=5).mean()
    tr, te = train_test_split(np.arange(len(df)), test_size=0.1, random_state=42, stratify=y)
    clf.fit(X.iloc[tr], y[tr]); a_te = clf.score(X.iloc[te], y[te])
    print(f"  DRIVER tree: baseline {base:.3f} | CV {cv:.3f} (lift +{cv-base:.3f}) | test {a_te:.3f}")

    rules = leaf_rules(clf, inputs, cls_names)
    pure = [r for r in rules if r["purity"] >= PURITY]
    with open(os.path.join(out, "rules.txt"), "w", encoding="utf-8") as f:
        f.write(f"def {name} ({metric_col}) k*={best_k} baseline {base:.3f} CV {cv:.3f} test {a_te:.3f}\n")
        for i in range(best_k):
            xi = x[y == i]
            f.write(f"  D{i}: n={len(xi)} range[{xi.min():.3f},{xi.max():.3f}] mean {xi.mean():.3f}\n")
        f.write("\nconfident rules:\n")
        for r in sorted(pure, key=lambda r: -r["purity"]):
            f.write(f"  [{r['purity']:.0%}, n={r['n']}] {r['class']}  IF {r['rule']}\n")

    fig, ax = plt.subplots(figsize=(16, 9))
    plot_tree(clf, feature_names=inputs, class_names=cls_names, filled=True, rounded=True,
              proportion=True, impurity=True, precision=1, fontsize=9, ax=ax)
    ax.set_title(f"Decentralisation def {name} ({metric_col}) — inputs → level "
                 f"(k*={best_k}, baseline {base:.2f}, CV {cv:.2f}, test {a_te:.2f})", fontsize=14)
    fig.savefig(os.path.join(out, "driver_tree.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    return {"def": name, "tag": tag, "k": best_k, "baseline": base, "cv": cv,
            "lift": cv - base, "test": a_te}


def main():
    C.apply_joule_style()
    os.makedirs(OUT, exist_ok=True)
    feat = pd.read_csv(C.FEATURES_CSV)
    dm = load_metrics()
    rob_ids, *_ = robust_run_ids(ROBUST_SCENARIO)

    subsets = {
        "all": (feat, dm),
        "robust": (feat[feat["run"].isin(rob_ids)], dm[dm["run"].isin(rob_ids)]),
    }
    defs = [("A_smallprod_over_totaldemand", "defA"),
            ("B_smallprod_over_smalldemand", "defB")]
    res = []
    for name, col in defs:
        for tag, (fsub, dsub) in subsets.items():
            res.append(run_def(fsub.reset_index(drop=True), dsub.reset_index(drop=True),
                               name, col, tag))
    print("\n" + "=" * 70)
    print(pd.DataFrame(res).to_string(index=False))
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree import plot_tree


def plot_decision_tree(clf, feature_names, class_names, title, save_path, export_format):
    """Plot and save a sklearn decision tree visualization."""
    plt.figure(figsize=(40, 20))
    plot_tree(clf, feature_names=feature_names, class_names=class_names,
              filled=True, rounded=True, fontsize=13)
    plt.title(title, fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path, **({"dpi": 300} if export_format == "png" else {}), bbox_inches="tight")
    print(f"\nTree plot saved as: {save_path}")
    plt.close()


def plot_rf_comparison(clf, rf, feature_names, dt_title, rf_title, save_path, export_format):
    """Plot and save a side-by-side Decision Tree vs Random Forest feature importance chart."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

    dt_importances = pd.Series(clf.feature_importances_, index=feature_names).sort_values(ascending=False)
    dt_importances[:10].plot(kind='barh', ax=ax1, color='steelblue')
    ax1.set_xlabel('Importance', fontsize=12)
    ax1.set_title(dt_title, fontsize=14, fontweight='bold')
    ax1.invert_yaxis()
    ax1.grid(axis='x', alpha=0.3)

    rf_importances = pd.Series(rf.feature_importances_, index=feature_names).sort_values(ascending=False)
    rf_importances[:10].plot(kind='barh', ax=ax2, color='forestgreen')
    ax2.set_xlabel('Importance', fontsize=12)
    ax2.set_title(rf_title, fontsize=14, fontweight='bold')
    ax2.invert_yaxis()
    ax2.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, **({"dpi": 300} if export_format == "png" else {}), bbox_inches="tight")
    print(f"\nRandom Forest comparison plot saved as: {save_path}")
    plt.close()
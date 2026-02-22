from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA


def run_pca_analysis(
    X_train: np.ndarray,
    X_test: np.ndarray,
    feature_names: List[str],
    n_components: int = 3,
    output_dir: str = "results",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    components = pca.components_
    explained_variance_ratio = pca.explained_variance_ratio_

    corr = np.corrcoef(X_train, X_train_pca, rowvar=False)
    n_features = X_train.shape[1]
    corr_matrix = corr[:n_features, n_features:]
    corr_df = pd.DataFrame(
        corr_matrix,
        index=feature_names,
        columns=[f"PC{i + 1}" for i in range(n_components)],
    )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Scree plot
    plt.figure(figsize=(8, 5))
    plt.plot(
        range(1, n_components + 1),
        explained_variance_ratio,
        marker="o",
        linestyle="-",
        color="tab:blue",
    )
    plt.title("PCA Scree Plot")
    plt.xlabel("Principal Component")
    plt.ylabel("Explained Variance Ratio")
    plt.xticks(range(1, n_components + 1))
    plt.tight_layout()
    plt.savefig(output_path / "pca_scree_plot.png")
    plt.close()

    # Correlation heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(
        corr_df,
        annot=False,
        cmap="coolwarm",
        center=0,
        cbar_kws={"label": "Correlation"},
    )
    plt.title("Correlation: Original Features vs PCA Components")
    plt.tight_layout()
    plt.savefig(output_path / "pca_feature_pc_correlation.png")
    plt.close()

    return X_train_pca, X_test_pca, components, explained_variance_ratio, corr_df

from __future__ import annotations

from typing import Tuple

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


def run_lda_analysis(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    n_components: int = 2,
) -> Tuple[np.ndarray, np.ndarray, LinearDiscriminantAnalysis, int]:
    n_classes = len(np.unique(y_train))
    max_components = min(n_components, n_classes - 1)
    if max_components < 1:
        raise ValueError("LDA requires at least two classes for dimensionality reduction.")

    lda = LinearDiscriminantAnalysis(n_components=max_components)
    X_train_lda = lda.fit_transform(X_train, y_train)
    X_test_lda = lda.transform(X_test)
    return X_train_lda, X_test_lda, lda, max_components

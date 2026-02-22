from __future__ import annotations

from typing import List, Tuple

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


def forward_wrapper_selection(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    k: int = 9,
    n_features: int = 3,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[List[int], List[str]]:
    X = np.asarray(X)
    y = np.asarray(y)

    selected_features: List[int] = []
    remaining_features = list(range(X.shape[1]))

    while len(selected_features) < n_features and remaining_features:
        best_feature = None
        best_accuracy = -1.0

        for feature_idx in remaining_features:
            candidate_features = selected_features + [feature_idx]
            X_candidate = X[:, candidate_features]

            X_train, X_val, y_train, y_val = train_test_split(
                X_candidate,
                y,
                test_size=test_size,
                random_state=random_state,
                stratify=y,
            )

            model = KNeighborsClassifier(n_neighbors=k)
            model.fit(X_train, y_train)
            preds = model.predict(X_val)
            acc = accuracy_score(y_val, preds)

            if acc > best_accuracy:
                best_accuracy = acc
                best_feature = feature_idx

        if best_feature is None:
            break

        selected_features.append(best_feature)
        remaining_features.remove(best_feature)

    selected_feature_names = [feature_names[i] for i in selected_features]
    return selected_features, selected_feature_names

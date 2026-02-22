from __future__ import annotations

from typing import Tuple, Union

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier


def evaluate_knn(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    k: int = 9,
    return_details: bool = False,
) -> Union[float, Tuple[float, np.ndarray, str]]:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    accuracy = accuracy_score(y_test, preds)
    conf_matrix = confusion_matrix(y_test, preds)
    report = classification_report(y_test, preds)

    if return_details:
        return accuracy, conf_matrix, report
    return accuracy

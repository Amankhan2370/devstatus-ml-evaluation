from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler


def load_and_preprocess_data(
    file_path: str, target_column: str = "development_status"
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    df = pd.read_csv(file_path)
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataset.")

    df = df.dropna(subset=[target_column]).reset_index(drop=True)
    feature_df = df.drop(columns=[target_column])

    # Mean imputation for numeric features
    feature_df = feature_df.fillna(feature_df.mean(numeric_only=True))

    if feature_df.isnull().any().any():
        raise ValueError("Missing values remain after mean imputation.")

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df[target_column].astype(str))

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(feature_df.values)

    feature_names = feature_df.columns.tolist()
    return X_scaled, y, feature_names

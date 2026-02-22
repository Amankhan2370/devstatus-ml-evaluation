from __future__ import annotations

from pathlib import Path

import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

from forward_wrapper import forward_wrapper_selection
from knn_evaluation import evaluate_knn
from lda_analysis import run_lda_analysis
from pca_analysis import run_pca_analysis
from preprocessing import load_and_preprocess_data


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    data_path = project_root / "data" / "world_data.csv"
    results_dir = project_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    X_scaled, y, feature_names = load_and_preprocess_data(str(data_path))

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    selected_indices, selected_names = forward_wrapper_selection(
        X_train, y_train, feature_names
    )
    X_train_selected = X_train[:, selected_indices]
    X_test_selected = X_test[:, selected_indices]
    accuracy_forward = evaluate_knn(X_train_selected, X_test_selected, y_train, y_test)

    X_train_pca, X_test_pca, _, _, _ = run_pca_analysis(
        X_train, X_test, feature_names, output_dir=str(results_dir)
    )
    accuracy_pca = evaluate_knn(X_train_pca, X_test_pca, y_train, y_test)

    X_train_lda, X_test_lda, _, lda_components = run_lda_analysis(
        X_train, X_test, y_train
    )
    accuracy_lda = evaluate_knn(X_train_lda, X_test_lda, y_train, y_test)

    results_df = pd.DataFrame(
        [
            {
                "method": "Forward Wrapper (3 features)",
                "accuracy": accuracy_forward,
                "selected_features": ", ".join(selected_names),
            },
            {"method": "PCA (3 components)", "accuracy": accuracy_pca},
            {"method": f"LDA ({lda_components} components)", "accuracy": accuracy_lda},
        ]
    )

    print("\nAccuracy Comparison")
    print(results_df[["method", "accuracy"]].to_string(index=False))

    results_path = results_dir / "accuracy_results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"\nSaved results to: {results_path}")

    # Benchmark bar chart
    plt.figure(figsize=(8, 4.5))
    plt.bar(results_df["method"], results_df["accuracy"], color="tab:blue")
    plt.ylabel("Accuracy")
    plt.title("KNN Accuracy by Feature Strategy")
    plt.xticks(rotation=20, ha="right")
    plt.ylim(0, 1)
    plt.tight_layout()
    benchmark_path = results_dir / "accuracy_benchmark.png"
    plt.savefig(benchmark_path)
    plt.close()
    print(f"Saved benchmark plot to: {benchmark_path}")


if __name__ == "__main__":
    main()

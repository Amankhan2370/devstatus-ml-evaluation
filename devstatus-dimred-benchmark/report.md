# Dimensionality Reduction Benchmark Report

## Wrapper Method Theory
Wrapper methods evaluate subsets of features by training a predictive model and selecting the subset that maximizes a chosen metric. Forward selection is a greedy wrapper strategy that starts with zero features and iteratively adds the feature that yields the best validation performance. Although computationally more expensive than filter methods, wrapper methods often produce subsets that are directly optimized for a given model and objective.

## PCA Mathematical Intuition
Principal Component Analysis (PCA) is an orthogonal linear transformation that projects data onto directions of maximum variance. Mathematically, PCA finds the eigenvectors of the covariance matrix of the data. The leading eigenvectors define the principal components, and their eigenvalues quantify the explained variance. PCA is unsupervised, so it does not consider class labels during projection.

## LDA Class-Separation Intuition
Linear Discriminant Analysis (LDA) seeks projections that maximize between-class variance while minimizing within-class variance. It explicitly incorporates class labels, producing directions that are most discriminative for classification. In binary classification, LDA yields at most one discriminant direction, but multi-component representations can be used when more than two classes exist or for standardized pipelines.

## Why LDA May Outperform PCA for Classification
Because LDA is supervised, its projections prioritize class separability rather than overall variance. PCA can emphasize directions with large variance that are not necessarily relevant for distinguishing classes. As a result, LDA often yields higher classification accuracy, especially when class separation is subtle and aligns with low-variance directions in the original feature space.

## Interpretation of Results
The benchmark compares accuracy across:
- Forward Wrapper selection on original features (3 features).
- PCA projection into 3 components.
- LDA projection into 2 components.

The accuracy results stored in `results/accuracy_results.csv` quantify the impact of each representation on KNN performance. Forward Wrapper is expected to perform competitively when a small subset of original features strongly correlates with the class label. PCA is expected to perform well when the data is highly correlated, while LDA is expected to excel when class boundaries are well-defined.

## Practical Implications
For downstream decision-making, feature selection can improve interpretability by retaining original variables, while PCA offers compact representations at the cost of interpretability. LDA provides strong performance for classification tasks but can be sensitive to class imbalance or violations of its linear separability assumptions.

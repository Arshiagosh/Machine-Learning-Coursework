# HW4: Feature selection and dimensionality reduction

SelectKBest, RFE and random-forest importances on the Wisconsin breast-cancer data, then PCA and t-SNE projections. The written part covers PCA by hand, PCA vs. Fisher LDA, forward/backward search, the knapsack view of feature selection, and kernel PCA.

## Files

- [`report_en.pdf`](report_en.pdf): English summary (LaTeX source next to it)
- [`breast_cancer_features.ipynb`](breast_cancer_features.ipynb): notebook (re-run for this repository)
- `figures/`: plots from the notebook

Data: `breast-cancer.csv`, see [`data/README.md`](../data/README.md).

**Changes for this repository:** the submitted notebook failed in five cells. I fixed it so it runs: the `id` column is no longer used as a feature, feature names are taken from `X` instead of `df`, and the scatter plots colour by `diagnosis == 'M'`. PCA still runs on unscaled features, as submitted, so the area features dominate (99.8 % variance in two components).

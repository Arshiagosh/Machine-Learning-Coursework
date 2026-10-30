# Datasets

The datasets are not stored in this repository. Each notebook expects its file next to it
(same folder), under the file name given below.

| Used in | Dataset | File name | Where to get it |
|---|---|---|---|
| HW2 | Abalone (UCI) | downloaded by the notebook | <https://archive.ics.uci.edu/dataset/1/abalone> |
| HW2 | Iris | built into scikit-learn | `sklearn.datasets.load_iris` |
| HW3 | IBM HR Analytics Employee Attrition (fictional data) | `dataset.csv`, `test.csv` | <https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset>; `test.csv` was a held-out split handed out with the assignment, so make your own split from the Kaggle file |
| HW4 | Breast Cancer Wisconsin (Diagnostic) | `breast-cancer.csv` | <https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic> (the CSV with `id`, `diagnosis` and 30 features) |
| HW5 | Statlog (Landsat Satellite), standardised | `satimage.csv` | <https://archive.ics.uci.edu/dataset/146/statlog+landsat+satellite> (columns `0`..`35` scaled to zero mean, unit variance, plus `classes`) |
| HW6 | MNIST, CIFAR-10 | downloaded by torchvision | `torchvision.datasets` |
| HW7 | California Housing | downloaded by scikit-learn | `sklearn.datasets.fetch_california_housing` |
| Sudoku | Printed digits (Kaggle) | `Sudoku-Solver-from-a-Photo/assets/assets/0..9/` | <https://www.kaggle.com/datasets/kshitijdhama/printed-digits-dataset> (only needed to retrain the digit classifiers) |

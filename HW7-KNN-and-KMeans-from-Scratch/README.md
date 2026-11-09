# HW7: KNN and K-means from scratch, EM

KNN from scratch matches scikit-learn exactly (test accuracy 0.762, K = 5). K-means from scratch matches scikit-learn's elbow and silhouette curves on California Housing. The written part covers leave-one-out choice of K, K-means convergence and the scatter decomposition, and EM for a two-coin mixture.

## Files

- [`report_en.pdf`](report_en.pdf): English summary (LaTeX source next to it)
- [`knn-from-scratch.ipynb`](knn-from-scratch.ipynb), [`kmeans-from-scratch.ipynb`](kmeans-from-scratch.ipynb): notebooks (re-run for this repository)
- `figures/`: plots from the notebooks

Data: the KNN notebook reads the HW3 attrition file as `dataset.csv` (see [`data/README.md`](../data/README.md)); California Housing is downloaded by scikit-learn.

**Changes for this repository:** the submitted KNN notebook never loaded its data and was missing two imports. I added `df = pd.read_csv('dataset.csv')` and the `Counter` and `KNeighborsClassifier` imports to the first cell; the rest is unchanged.

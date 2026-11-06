# HW5: Support vector machines

A soft-margin SVM solved as a dual QP with `cvxopt`, linear and RBF, written from scratch (test accuracy 0.961, classes 4 vs. 5 of satimage). Then a tuned multiclass RBF `SVC` (0.929) and a comparison of linear, RBF, polynomial and sigmoid kernels. The written part covers the L2 soft-margin dual, a two-point max-margin problem and kernel closure properties.

## Files

- [`report_en.pdf`](report_en.pdf): English summary (LaTeX source next to it)
- [`svm.ipynb`](svm.ipynb): notebook (re-run for this repository)
- `figures/`: confusion matrices from the notebook

Data: `satimage.csv`, see [`data/README.md`](../data/README.md).

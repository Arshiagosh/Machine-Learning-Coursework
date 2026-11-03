# HW3: KNN and bagging for employee attrition

A KNN classifier written from scratch and tuned over K (best K = 9, test accuracy 0.830), compared with bagged KNN (0.827) on the IBM HR attrition data. Both have poor recall on the minority class. The written part covers a naive-Bayes divide-by-zero example, a perceptron run by hand, and Bayes' rule for a rare-disease test.

## Files

- [`report_en.pdf`](report_en.pdf): English summary (LaTeX source next to it)
- [`employee_attrition.ipynb`](employee_attrition.ipynb): notebook (re-run for this repository)
- `figures/`: plots from the notebook

Data: `dataset.csv` and `test.csv`, see [`data/README.md`](../data/README.md).

**Change for this repository:** the notebook dropped three constant columns from `test.csv` before loading it (cells had been run out of order). That line now runs right after `test.csv` is loaded, so the notebook runs top to bottom.

# Classification Metrics — Reference Guide

---

## Confusion Matrix Basics

```
                  Predicted Positive    Predicted Negative
Actual Positive        TP                    FN
Actual Negative        FP                    TN
```

- **TP** — correctly predicted positive
- **TN** — correctly predicted negative
- **FP** — predicted positive, actually negative (Type I error)
- **FN** — predicted negative, actually positive (Type II error)

---

## Metrics

### Predictive Value (column-based — what your predictions are worth)

| Metric | Formula | Meaning |
|---|---|---|
| **PPV** (Precision) | TP / (TP + FP) | Of all predicted positives, how many are actually positive |
| **NPV** | TN / (TN + FN) | Of all predicted negatives, how many are actually negative |
| **FDR** | FP / (FP + TP) | Of all predicted positives, how many are wrong — `1 - PPV` |
| **FOR** | FN / (FN + TN) | Of all predicted negatives, how many are wrong — `1 - NPV` |

### Rates (row-based — how well you cover actual classes)

| Metric | Formula | Meaning |
|---|---|---|
| **TPR** (Recall / Sensitivity) | TP / (TP + FN) | Of all actual positives, how many did you catch |
| **TNR** (Specificity) | TN / (TN + FP) | Of all actual negatives, how many did you correctly reject |
| **FPR** | FP / (FP + TN) | Of all actual negatives, how many did you wrongly flag — `1 - TNR` |
| **FNR** | FN / (TP + FN) | Of all actual positives, how many did you miss — `1 - TPR` |

### Overall

| Metric | Formula | Meaning |
|---|---|---|
| **Accuracy** | (TP + TN) / (TP + TN + FP + FN) | Overall fraction correct — misleading on imbalanced datasets |

---

## Precision, Recall, and F1

**Precision** (PPV) and **Recall** (TPR) trade off against each other:

- Increase threshold → higher precision, lower recall (fewer but more confident positives)
- Decrease threshold → higher recall, lower precision (catch more positives but more false alarms)

**F1 Score** is the harmonic mean of precision and recall — useful when you want a single balanced metric:

```
F1 = 2 * (Precision * Recall) / (Precision + Recall)
```

The harmonic mean penalises large gaps between precision and recall more than a simple average would.

---

## MCC — Matthews Correlation Coefficient

```
MCC = (TP*TN - FP*FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))
```

- Range: **-1 to +1** (+1 = perfect, 0 = random, -1 = perfectly wrong)
- Considered one of the most reliable single metrics for binary classification
- Unlike accuracy and F1, MCC accounts for all four cells of the confusion matrix
- Works well even on **imbalanced datasets**

---

## Multi-class Averaging

When you have more than two classes, metrics like precision, recall, and F1 are computed per class and then averaged. Three averaging strategies:

### Macro
Compute the metric for each class independently, then take the **unweighted average**.

```
Macro F1 = mean(F1_class1, F1_class2, F1_class3)
```

- Treats all classes equally regardless of size
- Use when **every class matters equally**, even minority ones

### Weighted
Compute the metric for each class, then average weighted by the **number of true samples** in each class.

```
Weighted F1 = sum(F1_class_i * support_i) / total_samples
```

- Accounts for class imbalance
- Use when **larger classes matter more** or you want an overall performance picture

### Micro
Aggregate all TP, FP, FN across classes first, then compute the metric once globally.

```
Micro Precision = sum(TP_i) / sum(TP_i + FP_i)
```

- Dominated by the most frequent class
- Equivalent to accuracy for precision/recall in multi-class settings

### When to use which

| Scenario | Recommended |
|---|---|
| Balanced classes | Macro or Micro (similar results) |
| Imbalanced classes, minority class matters | Macro |
| Imbalanced classes, want overall picture | Weighted |
| Single summary close to accuracy | Micro |

---

## Quick Reference — Which Metric for Which Problem

| Problem | Metric to prioritise |
|---|---|
| Imbalanced dataset | MCC, Macro F1 |
| Cost of false positives is high (e.g. spam filter) | Precision (PPV) |
| Cost of false negatives is high (e.g. disease screening) | Recall (TPR) |
| Balance between precision and recall | F1 |
| General overview | Accuracy (only if balanced) |
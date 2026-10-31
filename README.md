# M5+ Solar Flare Forecasting

Predict whether a solar active region will produce an M5+ flare in the next 24 hours, using only magnetic field measurements from NASA's Solar Dynamics Observatory (SDO).

M5+ flares are the class responsible for events like the October 2003 flare that disrupted GPS across North America and caused power outages in Sweden — a 24-hour warning is operationally useful.

Capstone project for Astronomy 101 (Equinox, IIT Guwahati) — Computational Module.

## Dataset

- Source: NASA SDO, Jan 2023 – May 2025, ~12-minute cadence snapshots per active region.
- Label: `Flare` (M5+ in next 24h: yes/no).
- Class imbalance: ~11:1 (negative:positive) in training data, worse in test data.
- Not included in this repo — see `data/README.md`.

## Approach

- **Group-aware split** (`GroupShuffleSplit` / `StratifiedGroupKFold` on active region ID) instead of a random row split — rows are dense time-series snapshots of the same region, so a random split leaks near-duplicate rows between train and test.
- Identifier columns (`NOAA_AR`, `HARPNUM`) excluded from features.
- Imputation and SMOTE fit only inside pipelines, on training folds.
- 5 tree-based models compared (Decision Tree, Random Forest, AdaBoost, Gradient Boosting, XGBoost), tuned with `RandomizedSearchCV`, all using consistent sample-weight-based imbalance handling.
- Model selected by cross-validated F1 on training data only; test set touched once, at the end.
- Metrics: Precision, Recall, F1, ROC-AUC, PR-AUC, TSS, HSS (standard in flare-forecasting literature; accuracy/ROC-AUC alone are close to meaningless at this prevalence).

## Results (held-out test set)

Selected model: **AdaBoost** (chosen by CV F1 on training data; statistically tied with Gradient Boosting — see notebook).

| Model              | Accuracy | Precision | Recall | F1    | ROC-AUC | PR-AUC | TSS   | HSS   |
|--------------------|---------:|----------:|-------:|------:|--------:|-------:|------:|------:|
| Decision Tree      | 0.784    | 0.293     | 0.460  | 0.358 | 0.722   | 0.293  | 0.293 | 0.236 |
| Random Forest      | 0.791    | 0.321     | 0.528  | 0.399 | 0.825   | 0.330  | 0.359 | 0.282 |
| **AdaBoost**       | 0.730    | 0.309     | **0.858** | 0.454 | 0.874 | 0.586  | 0.568 | 0.324 |
| Gradient Boosting  | 0.750    | 0.321     | 0.814  | 0.461 | 0.844   | 0.404  | 0.554 | 0.336 |
| XGBoost            | 0.781    | 0.308     | 0.541  | 0.393 | 0.810   | 0.379  | 0.358 | 0.271 |

AdaBoost final confusion matrix: TP 2309, FN 382, FP 5166, TN 12669 (2691 positives / 17835 negatives in test).

## Key findings

- Fixing the row-wise leakage dropped scores from a (false) ~0.99 F1 / 1.0 ROC-AUC to the numbers above — the honest picture of this task's difficulty.
- AdaBoost and Gradient Boosting are statistically tied in cross-validation (F1 means within 1 std of each other); AdaBoost was favored for its higher recall.
- AdaBoost's predicted probabilities are poorly calibrated (known property of SAMME boosting) — see the reliability curve in the notebook. Use `CalibratedClassifierCV` if probability estimates matter, not just classification.
- Precision is low (~0.31) by design: missing a flare is costlier than a false alarm, so the model is tuned toward recall.

## Repo structure

```
notebooks/   analysis notebook (EDA, feature engineering, modeling, evaluation)
data/        not included — see data/README.md
reports/     exported figures (confusion matrix, ROC, PR, calibration curves)
```

## Run it

```bash
pip install -r requirements.txt
# place training_data_M5_24.csv in data/ (see data/README.md)
jupyter notebook notebooks/M5_Flare_Problem.ipynb
```

## License

MIT — see `LICENSE`.

# Credit Risk — Probability of Default Model

End-to-end credit risk project: data-quality audit, leak-free preprocessing, model comparison, profit-based decision threshold, calibration and explainability.

## Results (test set, 6,482 applications)

| | Logistic Regression | XGBoost |
|---|---|---|
| ROC-AUC | 0.870 | 0.953 |
| Gini / KS | 0.74 / 0.59 | 0.91 / 0.78 |
| Brier score (no-skill: 0.171) | 0.107 | 0.049 |
| Defaulters caught / precision of declines | 83% / 48% | 86% / 72% |
| Simulated profit (approve-all: −4.32M) | +2.13M | +3.34M |

**Recommendation:** XGBoost. It ranks risk better, its probabilities are well calibrated, and at the profit-optimal threshold (0.18) it declines far fewer good customers. Its main advantage comes from a non-linear effect: default risk jumps once a loan exceeds about 30% of income.

**Scope of use:** `loan_grade` and `loan_int_rate` are assigned by the lender itself, so the full models apply to decisions taken after grading and pricing. Without those two features, cross-validated AUC falls to 0.817 (Logistic Regression) and 0.903 (XGBoost); the reduced XGBoost is a credible pre-approval model.

## What the notebook covers

1. **Data-quality audit:** single-column and cross-column checks with real-world rules; 165 duplicates and 7 impossible rows removed; an inconsistent derived field recomputed.
2. **Leak-free methodology:** only rule-based fixes before the split; exploratory analysis on the training set only; all learned transformations inside sklearn `Pipeline`s (including a custom grade-conditional imputer).
3. **Model selection:** 5-fold stratified cross-validation with hyperparameter search; the test set is used once. A reduced variant without the lender's own grade and rate measures how much performance is inherited from the lender's assessment.
4. **Decision policy, separate from the model:** threshold chosen by simulated portfolio profit under explicit assumptions (LGD 60%, with sensitivity at 40% and 80%).
5. **Calibration:** calibration curves and Brier score, since a PD model must produce trustworthy probabilities.
6. **Explainability:** Logistic Regression coefficients (with a multicollinearity caveat) and SHAP for XGBoost, including individual decision explanations.

7. **Production view:** what would change in a live setting (out-of-time validation, PSI drift monitoring, backtesting, recalibration, reject inference, governance).

## Data

[Credit Risk Dataset](https://www.kaggle.com/datasets/laotse/credit-risk-dataset) (Kaggle, `laotse/credit-risk-dataset`), 32,581 loan applications. Downloaded automatically in the notebook with `kagglehub`.

## How to run

Open `credit_risk_analysis_v2.ipynb` in Google Colab and run all cells. Dependencies: `pandas`, `numpy`, `scikit-learn`, `xgboost`, `shap`, `statsmodels`, `scipy`, `matplotlib`, `seaborn`, `kagglehub`.

## Limitations

Educational dataset without application dates (no out-of-time validation); the profit simulation relies on stated assumptions about loss given default and interest income.

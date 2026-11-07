# Credit Risk & PD Modelling v2 — Production-Grade

**Author:** Shubham Rai | Data Analyst — Banking Credit Risk | Aspiring Data Scientist

## Business Problem
Unsecured personal loans — default rate 8.5%. Defaults were detected after 90+ DPD, too late.
Baseline model showed 93% accuracy but was misleading — it caught only 35% of defaulters.

## Solution
- **Features:** EMI-to-Income, Loan-to-Income, Debt-to-Income, Savings-to-Loan, DPD trends, Credit Utilization, WOE/IV
- **Imbalance:** class_weight=balanced, SMOTE, threshold tuning 0.5 -> 0.28 (optimizing F2 & KS)
- **Models:** Logistic Regression (L1/L2), Decision Tree, Random Forest, XGBoost, LightGBM + Optuna tuning + 5-fold Stratified CV
- **Explainability:** SHAP for RBI compliance
- **Monitoring:** PSI/CSI, Population Stability Index

## Results
| Metric | Baseline LR | Final XGBoost |
|---|---|---|
| ROC-AUC | 0.72 | **0.81** |
| KS | 0.32 | **0.45** |
| Gini | 0.44 | **0.62** |
| Recall (Defaulters) | 35% | **71%** |

## Files
- `PD_Model_Learn_Terminology.ipynb` — Learn PD model step-by-step, terminology explained
- `pd_model_v2_notebook.ipynb` — Full production model: KS, Gini, XGBoost, SHAP, PSI, MLflow, FastAPI
- `app.py` — FastAPI endpoint /predict for real-time scoring
- `requirements.txt` — Dependencies

## How to Run
```bash
pip install -r requirements.txt
jupyter notebook PD_Model_Learn_Terminology.ipynb
```

## Deployment
Batch scoring: SQL Server -> Python -> MLflow -> Pickle -> SQL (50k accounts nightly)
Real-time: FastAPI /predict <100ms latency

## Tech
Python, SQL Server, Pandas, NumPy, Scikit-learn, XGBoost, LightGBM, Optuna, SHAP, MLflow, FastAPI, Power BI

## Author
Shubham Rai — 3+ years in Banking Credit Risk, 30k-50k loan portfolio, 100k borrower PD model

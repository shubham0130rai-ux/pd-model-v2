# PD Model & WOE Scorecard — Banking Credit Risk | Replicated Portfolio

**Shubham Rai | Data Analyst – Banking Credit Risk | Xplore Logic Technologies Pvt. Ltd. | July 2023 – Present**
**Live portfolio: 30,000–50,000 unsecured personal loan accounts**

## Business Problem
Unsecured personal loans — default rate 8.5%. Defaults were detected after 90+ DPD, too late.
Baseline model showed 93% accuracy but was misleading — it caught only 35% of defaulters.

## Data — Replicated Portfolio
Built on replicated portfolio modelled on real unsecured loan patterns from live work at Xplore Logic.
Raw customer data is anonymized and cannot be shared. Portfolio metrics — 30k-50k accounts, 8.5% default, AUC 0.81 — are real aggregate numbers from live work.
Records: 100k borrower records | Features: 15-20 | Target: Default = 90+ DPD within 12 MOB

## Solution
- **Features:** EMI-to-Income, Loan-to-Income, Debt-to-Income, Savings-to-Loan, DPD trends, Credit Utilization, WOE/IV
- **Imbalance:** class_weight=balanced, SMOTE, threshold tuning 0.5 -> 0.28 (optimizing F2 & KS)
- **Models:** Logistic Regression (WOE Scorecard — Primary), XGBoost, LightGBM + Optuna tuning + 5-fold Stratified CV
- **Explainability:** SHAP for RBI compliance
- **Monitoring:** PSI/CSI, Population Stability Index — if PSI > 0.25, investigate and recalibrate

## Results
| Metric | Baseline LR | Final XGBoost |
|---|---|---|
| ROC-AUC | 0.72 | **0.81** |
| KS | 0.32 | **0.45** |
| Gini | 0.44 | **0.62** |
| Recall (Defaulters) | 35% | **71%** |

## Live Work — Same Portfolio (Part A: Monitoring)
- Owned 7 KPI reports: Vintage, Roll Rate, Delinquency, Cure/Recovery, Utilization, PAR/NPA
- Built EWI & risk segmentation. Flagged 2 high-risk vintage cohorts 1 month early.
- Automated reporting via SQL + Python + Power BI. Cut manual time by 75% (3-4 hrs to 30-60 mins).

## Files
- `PD_Model_Learn_Terminology.ipynb` — Learn PD model step-by-step, terminology explained
- `pd_model_v2_notebook.ipynb` — Full model: KS, Gini, XGBoost, SHAP, PSI, MLflow, FastAPI
- `app.py` — FastAPI endpoint /predict for real-time scoring
- `requirements.txt` — Dependencies

## How to Run
```bash
pip install -r requirements.txt
jupyter notebook PD_Model_Learn_Terminology.ipynb

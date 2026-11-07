from fastapi import FastAPI
import pickle
import pandas as pd
app = FastAPI(title="PD Model API")
# Load model - train first using notebook to create pd_model_v2_xgb.pkl
# with open('pd_model_v2_xgb.pkl','rb') as f:
#     model = pickle.load(f)
@app.get("/")
def home():
    return {"message": "PD Model API — Shubham Rai | Use POST /predict"}
@app.post("/predict")
def predict(emi_to_income: float, dti: float, credit_score: float, dpd_max_6m: int, utilization: float, loan_to_income: float):
    # Example logic - replace with real model after training
    # proba = model.predict_proba([[emi_to_income, dti, credit_score, dpd_max_6m, utilization, loan_to_income]])[0][1]
    # Simplified rule for demo
    proba = 0.8 if (emi_to_income>0.5 and dpd_max_6m>30) else 0.15
    return {"pd_probability": float(proba), "risk_segment": "High" if proba>0.3 else "Low"}

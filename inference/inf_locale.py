import os
import joblib
import pandas as pd
import datetime

# 📍 Racine du projet (Churn_pred)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

MODEL_PATH = os.path.join(PROJECT_ROOT, "model", "xgb_churn_model.pkl")
FEATURES_PATH = os.path.join(PROJECT_ROOT, "model", "model_features.pkl")

model = joblib.load(MODEL_PATH)
features = joblib.load(FEATURES_PATH)

def predict(df):
    X = df[features]
    churn_proba = model.predict_proba(X)[:, 1]
    churn_label = (churn_proba >= 0.5).astype(int)

    return pd.DataFrame({
        "customer_id": df["customer_id"],
        "churn_probability": churn_proba,
        "churn_label": churn_label,
        "scoring_timestamp": datetime.datetime.utcnow()
    })

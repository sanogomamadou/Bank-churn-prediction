import joblib
import pandas as pd
import datetime
import os

MODEL_PATH = os.path.join(os.getenv("AZUREML_MODEL_DIR"), "xgb_churn_model.pkl")
FEATURES_PATH = os.path.join(os.getenv("AZUREML_MODEL_DIR"), "model_features.pkl")

def init():
    global model, features
    model = joblib.load(MODEL_PATH)
    features = joblib.load(FEATURES_PATH)

def run(mini_batch):
    print("📦 MINI BATCH CONTENT:", mini_batch)
    all_results = []

    for file_path in mini_batch:
        print("📂 Processing file:", file_path)

        df = pd.read_csv(file_path)
        print("🧾 Columns in input CSV:", list(df.columns))

        missing = set(features) - set(df.columns)

        if missing:
            print("❌ MISSING FEATURES DETECTED:")
            for col in missing:
                print(col)

            # 🔥 on retourne un DataFrame vide MAIS AVEC UNE COLONNE
            return pd.DataFrame({
                "error": [f"Missing features: {list(missing)}"]
            })

        X = df[features]
        churn_proba = model.predict_proba(X)[:, 1]
        churn_label = (churn_proba >= 0.5).astype(int)

        batch_df = pd.DataFrame({
            "customer_id": df["customer_id"],
            "churn_probability": churn_proba,
            "churn_label": churn_label,
            "scoring_timestamp": datetime.datetime.utcnow()
        })

        all_results.append(batch_df)

    # 🔥 OBLIGATOIRE : retourner un DataFrame
    return pd.concat(all_results, ignore_index=True)

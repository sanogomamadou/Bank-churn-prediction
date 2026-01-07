import streamlit as st
import pandas as pd
from inf_locale import predict


st.set_page_config(page_title="Churn Prediction", layout="wide")

st.title("📉 Customer Churn Prediction")
st.caption("Upload CSV → Prediction instantanée")

uploaded_file = st.file_uploader("Upload ton fichier clients (CSV)", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("Aperçu des données")
    st.dataframe(df.head())

    if st.button("🔮 Lancer la prédiction"):
        preds = predict(df)

        st.success("Prédiction terminée 🚀")
        st.dataframe(preds)

        # Sauvegarde pour Power BI
        preds.to_csv("predictions.csv", index=False)

        st.download_button(
            "⬇️ Télécharger les prédictions",
            preds.to_csv(index=False),
            "predictions.csv",
            "text/csv"
        )

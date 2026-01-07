import streamlit as st
import pandas as pd
from inf_locale import predict
#########
import yaml
from yaml.loader import SafeLoader
import streamlit_authenticator as stauth

with open("config.yaml") as file:
    config = yaml.load(file, Loader=SafeLoader)

authenticator = stauth.Authenticate(
    config["credentials"],
    config["cookie"]["name"],
    config["cookie"]["key"],
    config["cookie"]["expiry_days"]
)

name, authentication_status, username = authenticator.login("Login")

if authentication_status is False:
    st.error("❌ Identifiants incorrects")

if authentication_status is None:
    st.warning("👋 Merci de vous connecter")
    st.stop()

if authentication_status:
    authenticator.logout("Logout", "sidebar")
    st.sidebar.success(f"Connecté en tant que {name}")
####################


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

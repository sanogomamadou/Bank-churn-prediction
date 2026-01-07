import streamlit_authenticator as stauth

hashed_passwords = stauth.Hasher(["monmotdepassefort"]).generate()
print(hashed_passwords)

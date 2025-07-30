# auth.py
import streamlit as st
from appwrite_config import account
from appwrite.exception import AppwriteException

def login():
    st.subheader("🔐 Login")
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        try:
            session = account.create_email_session(email=email, password=password)
            st.session_state["user"] = email
            st.success("✅ Logged in successfully")
        except AppwriteException as e:
            st.error(f"Login failed: {e.message}")

def signup():
    st.subheader("🆕 Sign Up")
    email = st.text_input("Email", key="signup_email")
    password = st.text_input("Password", type="password", key="signup_pass")
    name = st.text_input("Name")
    if st.button("Sign Up"):
        try:
            account.create(email=email, password=password, name=name)
            st.success("✅ Account created. You can now log in.")
        except AppwriteException as e:
            st.error(f"Signup failed: {e.message}")

def logout():
    if st.button("Logout"):
        st.session_state.pop("user", None)
        st.success("🚪 Logged out")

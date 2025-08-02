# spam_app.py
from importlib.resources import open_text
import streamlit as st
import joblib
import string

model = joblib.load("spam_classifier.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

st.title("Spam Email Detector")
input_text = st.text_area("Enter email or SMS text:")

if st.button("Check"):
    cleaned = open_text(input_text)
    vec = vectorizer.transform([cleaned])
    result = model.predict(vec)[0]
    st.write("🚫 Spam" if result == 1 else "✅ Not Spam")

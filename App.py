import streamlit as st
import pickle

# Load saved model and vectorizer
with open("emotion_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("emotion_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# App title
st.title("Emotion Classification App")

# User input
text = st.text_area("Enter your text here")

# Predict button
if st.button("Predict Emotion"):
    if text.strip() == "":
        st.warning("Please enter some text")
    else:
        text_vector = vectorizer.transform([text])
        prediction = model.predict(text_vector)
        st.success(f"Predicted Emotion: {prediction[0]}")

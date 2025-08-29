import streamlit as st
from transformers import pipeline

# Load model
nlp = pipeline("sentiment-analysis")

st.title("🧠 Sentiment Analysis App")
st.write("Enter some text and find out if it's Positive or Negative!")

# Input text box
user_input = st.text_area("Your text:", "Enter text here")

if st.button("Analyze"):
    result = nlp(user_input)[0]
    st.write(f"**Prediction:** {result['label']}")
    st.write(f"**Confidence:** {result['score']:.2f}")

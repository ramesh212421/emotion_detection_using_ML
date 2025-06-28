import streamlit as st
import joblib

# Load the enhanced model and vectorizer
model = joblib.load('emotion_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Streamlit App UI
st.title(" Emotion Detection from Text")
st.markdown("Type any message below and let the AI detect the emotion for you!")

# Text input
user_input = st.text_area("✏️ Enter your text message here:")

# Prediction button
if st.button("Detect Emotion"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        # Vectorize input
        data = vectorizer.transform([user_input])
        # Predict emotion
        prediction = model.predict(data)
        st.success(f"🎯 Predicted Emotion: **{prediction[0]}**")

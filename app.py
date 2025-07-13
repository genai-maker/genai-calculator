import streamlit as st
from transformers import pipeline

# Set the model
model_name = "google/flan-t5-small"

# Load the model pipeline
@st.cache_resource
def load_model():
    return pipeline("text2text-generation", model=model_name)

generator = load_model()

# Streamlit UI
st.title("🧠 Ge

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
st.title("🧠 GenAI Calculator")
st.write("Enter a natural language math question and get the result using a language model.")

user_input = st.text_input("🔢 Ask a math question:", "What is 17 plus 25?")

if st.button("Calculate"):
    with st.spinner("Thinking..."):
        result = generator(user_input)[0]['generated_text']
        st.success(f"🧮 Result: {result}")

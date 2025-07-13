import streamlit as st
from transformers import pipeline

# Load lightweight GenAI model
model_name = "google/flan-t5-small"

# Cache the model to avoid reloading
@st.cache_resource
def load_model():
    return pipeline("text2text-generation", model=model_name)

generator = load_model()

# Streamlit UI
st.title("🧠 GenAI Calculator")
st.write("Ask a math question in natural language:")

user_input = st.text_input("🔢 Example: 'What is 12 plus 7?'")

if st.button("Calculate"):
    with st.spinner("Thinking..."):
        result = generator(user_input)[0]['generated_text']
        st.success(f"✅ Result: {result}")

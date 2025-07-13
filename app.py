import streamlit as st
import torch
from transformers import pipeline

# Load a lightweight text-generation model from Hugging Face
generator = pipeline("text-generation", model="EleutherAI/gpt-neo-1.3B")

# Set Streamlit app UI
st.set_page_config(page_title="GenAI Calculator", page_icon="🧮")
st.title("🧠 Free GenAI Calculator using Hugging Face")

# User input
user_input = st.text_input("Ask a math question or calculation (e.g., 'What is 25% of 80?'):")

# Run model when input is given
if user_input:
    with st.spinner("Thinking..."):
        # Generate response
        result = generator(user_input, max_length=100, do_sample=True, temperature=0.7)

        # Clean response (only show new part)
        generated_text = result[0]['generated_text']
        response = generated_text[len(user_input):].strip()

        # Show output
        st.subheader("🧮 Answer:")
        st.success(response if response else generated_text)

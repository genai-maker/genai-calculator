import streamlit as st
from transformers import pipeline

st.set_page_config(page_title="GenAI Calculator", page_icon="🧮")
st.title("🧠 Free GenAI Calculator (Cloud Version)")

generator = pipeline("text-generation", model="sshleifer/tiny-gpt2")

user_input = st.text_input("Ask a math question or calculation:")

if user_input:
    with st.spinner("Thinking..."):
        result = generator(user_input, max_length=100, do_sample=True, temperature=0.7)
        st.success(result[0]['generated_text'])

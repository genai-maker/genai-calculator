import streamlit as st
import torch   # 🔥 Important: Needed for transformer models
from transformers import pipeline

st.set_page_config(page_title="GenAI Calculator", page_icon="🧮")
st.title("🧠 Free GenAI Calculator (Cloud Version)")

# Use a lightweight model
generator = pipeline("text-generation", model="sshleifer/tiny-gpt2")

user_input = st.text_input("Ask a math question or calculation:")

if user_input:
    with st.spinner("Thinking..."):
        result = generator(user_input, max_length=100, do_sample=True, temperature=0.7)
        st.success(result[0]['generated_text'])

prompt = f"Answer this math question clearly and briefly: {user_input}"

inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=50)
answer = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

st.success(f"Answer: {answer}")


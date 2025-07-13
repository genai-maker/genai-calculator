import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load model and tokenizer
model_name = "google/flan-t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

st.set_page_config(page_title="GenAI Calculator", page_icon="🧮")
st.title("🤖 GenAI Calculator")
st.write("Enter a math question like `12 + 7` or `What is 45% of 200?`")

# Input
user_input = st.text_input("Your math question:")

if user_input:
    with st.spinner("Thinking..."):
        prompt = f"Answer this math question clearly and briefly: {user_input}"
        inputs = tokenizer(prompt, return_tensors="pt")
        outputs = model.generate(**inputs, max_new_tokens=50)
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

        st.success(f"**Answer: {answer}**")

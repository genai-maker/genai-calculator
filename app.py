import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Page settings
st.set_page_config(page_title="GenAI Calculator", page_icon="🧮")
st.title("🤖 GenAI Calculator")
st.markdown("Enter a math question like **12 + 7** or **What is 45% of 200?**")

# Load model
model_name = "mrm8488/t5-small-finetuned-math-gen"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# User input
user_input = st.text_input("Your math question:")

if user_input:
    with st.spinner("Calculating..."):
        prompt = f"solve: {user_input}"
        inputs = tokenizer(prompt, return_tensors="pt")
        outputs = model.generate(**inputs, max_new_tokens=50)
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    st.success(f"Answer: {answer}")

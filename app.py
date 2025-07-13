import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Set page title
st.set_page_config(page_title="🤖 GenAI Calculator")

# Title and instruction
st.title("🤖 GenAI Calculator")
st.write("Enter a math question like `12 + 7` or `What is 45% of 200?`")

# Load model and tokenizer
@st.cache_resource
def load_model():
    model_name = "mrm8488/t5-small-finetuned-math-gen"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_model()

# User input
user_input = st.text_input("Your math question:")

# Handle user input
if user_input:
    # Format the prompt
    prompt = f"Solve: {user_input} ="

    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt")

    # Generate output
    outputs = model.generate(**inputs, max_new_tokens=50)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    # Show result
    st.success(f"Answer: {answer}")

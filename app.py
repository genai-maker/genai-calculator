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
    model_name = "mrm8488/t5-small-finetuned-_

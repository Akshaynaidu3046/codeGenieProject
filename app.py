import streamlit as st
from transformers import pipeline

# Load the working model
generator = pipeline("text-generation", model="Salesforce/codegen-350M-multi")  # or your working model name

st.set_page_config(page_title="CodeGenie", layout="centered")

st.title("CodeGenie: AI-powered Code Generator")
st.markdown("Enter your code prompt below:")

user_input = st.text_input("Prompt", placeholder="e.g. give python code for prime numbers")

if st.button("Generate Code"):
    if user_input.strip() != "":
        with st.spinner("Generating code..."):
            try:
                result = generator(user_input, max_length=200)
                st.code(result[0]['generated_text'], language="python")
            except Exception as e:
                st.error(f"Error: {e}")
    else:
        st.warning("Please enter a prompt.")

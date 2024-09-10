import os
import streamlit as st
from pdf_processor import process_pdf
from chatbot import query_gpt
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="PDF Chatbot", layout="wide")

# Load the uploaded PDF
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

# If a PDF is uploaded, process it and store the text in session state
if uploaded_file is not None and 'pdf_text' not in st.session_state:
    st.session_state['pdf_text'] = process_pdf(uploaded_file)
    st.write("PDF has been processed. You can now ask questions.")

# If the PDF has been processed, display the chatbot
if 'pdf_text' in st.session_state:
    user_input = st.text_input("Ask a question")
    if user_input:
        # Query GPT-3.5-turbo with the PDF text and user question
        answer = query_gpt(st.session_state['pdf_text'], user_input)
        st.write(f"Answer: {answer}")

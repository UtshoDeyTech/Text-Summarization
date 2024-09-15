import streamlit as st
from pdf_processor import load_pdf, create_vector_db
from chatbot import create_rag_chain
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def main():
    st.set_page_config(page_title="PDF Chatbot", layout="wide")
    st.header("PDF Chatbot with Vector Database")

    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

    if uploaded_file:
        with st.spinner("Processing PDF..."):
            # Save the uploaded file temporarily
            with open("temp.pdf", "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            chunks = load_pdf("temp.pdf")
            vector_store = create_vector_db(chunks)
            st.success("PDF processed successfully!")

        rag_chain = create_rag_chain(vector_store)

        query = st.text_input("Ask a question about the PDF:")
        if query:
            with st.spinner("Generating answer..."):
                response = rag_chain.invoke(query)
                st.write(response)

        # Clean up the temporary file
        os.remove("temp.pdf")

if __name__ == "__main__":
    main()
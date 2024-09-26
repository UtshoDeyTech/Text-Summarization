import streamlit as st
import requests
import os
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

# FastAPI backend URL
BACKEND_URL = "http://localhost:5000"

# Set up OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_pdf_list():
    try:
        response = requests.get(f"{BACKEND_URL}/list_pdfs")
        response.raise_for_status()
        return response.json()["pdfs"]
    except requests.RequestException as e:
        st.error(f"Error fetching PDF list: {str(e)}")
        return []

def query_chromadb(query):
    try:
        response = requests.post(f"{BACKEND_URL}/search_chunks", json={"query": query, "n_results": 5})
        response.raise_for_status()
        return response.json()["results"]
    except requests.RequestException as e:
        st.error(f"Error querying ChromaDB: {str(e)}")
        return []

def get_openai_response(query, context):
    try:
        messages = [
            {"role": "system", "content": "You are a helpful assistant that answers questions based on the given context."},
            {"role": "user", "content": f"Context: {context}\n\nQuestion: {query}"}
        ]
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # Make sure this is the correct model name
            messages=messages,
            max_tokens=150
        )
        return response.choices[0].message.content, None
    except Exception as e:
        return None, str(e)

def generate_simple_response(query, chunks):
    if chunks:
        most_relevant_chunk = chunks[0]['text']
        return f"Based on the most relevant information found:\n\n{most_relevant_chunk}\n\nThis information seems most pertinent to your query: '{query}'. Please note that this is a direct extract and not an AI-generated response."
    return "I couldn't find any relevant information to answer your query."

def sync_chromadb():
    try:
        response = requests.post(f"{BACKEND_URL}/sync_chromadb")
        response.raise_for_status()
        result = response.json()
        st.success(f"Synchronization complete. {result['message']}")
        if result['deleted_pdfs']:
            st.info(f"Deleted PDFs: {', '.join(result['deleted_pdfs'])}")
    except requests.RequestException as e:
        st.error(f"Error during synchronization: {str(e)}")

def main():
    st.title("PDF Chatbot")

    # Add a button to sync ChromaDB
    if st.button("Sync ChromaDB with PDF folder"):
        sync_chromadb()

    # Check if any PDFs are uploaded
    pdfs = get_pdf_list()

    if not pdfs:
        st.warning("No PDFs available. Please upload a PDF using the FastAPI backend.")
        st.info("Make sure the FastAPI backend is running and accessible at " + BACKEND_URL)
    else:
        st.success(f"{len(pdfs)} PDF(s) available for querying.")

    # User input
    user_query = st.text_input("Ask a question about the uploaded PDFs:")

    if user_query:
        if not pdfs:
            st.error("No PDFs available to answer questions. Please upload a PDF first.")
        else:
            with st.spinner("Searching for relevant information..."):
                # Query ChromaDB using similarity search
                relevant_chunks = query_chromadb(user_query)

                if not relevant_chunks:
                    st.warning("No relevant information found in the uploaded PDFs.")
                else:
                    # Prepare context for OpenAI
                    context = "\n".join([chunk["text"] for chunk in relevant_chunks])

                    # Try to get response from OpenAI
                    with st.spinner("Generating response..."):
                        openai_response, error = get_openai_response(user_query, context)
                        
                        if openai_response:
                            st.subheader("AI-Generated Response:")
                            st.write(openai_response)
                        else:
                            st.warning(f"OpenAI API Error: {error}")
                            st.subheader("Fallback Response:")
                            fallback_response = generate_simple_response(user_query, relevant_chunks)
                            st.write(fallback_response)

                    # Display the relevant chunks used for context
                    st.subheader("Relevant Information:")
                    for i, chunk in enumerate(relevant_chunks, 1):
                        with st.expander(f"Chunk {i} (Similarity: {chunk['distance']:.4f})"):
                            st.write(chunk['text'])

if __name__ == "__main__":
    main()
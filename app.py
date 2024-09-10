import streamlit as st
import os
from functions import (
    basic_summarization_from_pdf,
    prompt_template_summarization_from_pdf,
    stuff_document_chain_summarization,
    map_reduce_summarization,
    map_reduce_custom_prompts,
    refine_chain_summarization
)

def check_file_type(file):
    file_extension = os.path.splitext(file.name)[1].lower()

    if file_extension in ['.pdf', '.doc', '.docx']:
        return "Hello! You uploaded a document.", True
    elif file_extension in ['.mp4', '.avi', '.mov', '.mkv', '.webm']:
        return "You uploaded a video file.", False
    elif file_extension in ['.mp3', '.wav', '.aac', '.flac']:
        return "You uploaded an audio file.", False
    elif file_extension in ['.jpg', '.jpeg', '.png', '.gif', '.bmp']:
        return "You uploaded an image file.", False
    else:
        return "Unsupported file type.", False



def main():
    st.title("File Upload Example")

    uploaded_file = st.file_uploader("Choose a file", type=['pdf', 'doc', 'docx', 'mp4', 'avi', 'mov', 'mkv', 'webm', 'mp3', 'wav', 'aac', 'flac', 'jpg', 'jpeg', 'png', 'gif', 'bmp'])
    
    if uploaded_file is not None:
        message, is_document = check_file_type(uploaded_file)
        st.write(message)

        if is_document:
            # Save the uploaded PDF file
            pdf_path = os.path.join("temp_dir", uploaded_file.name)
            os.makedirs("temp_dir", exist_ok=True)
            with open(pdf_path, 'wb') as f:
                f.write(uploaded_file.getbuffer())

            # Process the PDF file with all summarization functions
            st.write("### 1. Basic Summarization from PDF")
            st.write("**Description:** Extracts text from the PDF and generates a basic summary.")
            st.write(
        "- **When to Use:** Useful for straightforward summarization needs where a simple, concise summary is sufficient.\n"
        "- **Why:** This method is quick and easy, making it ideal for initial summarization when you need a general overview without complex requirements."
    )
            summary_basic = basic_summarization_from_pdf(pdf_path)
            st.write(summary_basic)
            
            st.write("### 2. Prompt Template Summarization (Translated to Hindi) from PDF")
            st.write("**Description:** Summarizes the PDF text and translates it to the specified language.")
            st.write(
        "- **When to Use:** Ideal when you need a summary in a specific language, especially for non-English content or for multilingual applications.\n"
        "- **Why:** This method combines summarization with translation, making it suitable for content that needs to be understood by speakers of different languages."
    )
            summary_prompt_template = prompt_template_summarization_from_pdf(pdf_path, language='Hindi')
            st.write(summary_prompt_template)
            
            st.write("### 3. StuffDocumentChain Summarization from PDF")
            st.write("**Description:** Summarizes the entire document in one pass using StuffDocumentChain.")
            st.write(
        "- **When to Use:** Effective for summarizing entire documents where a single, cohesive summary is needed.\n"
        "- **Why:** This method is straightforward and useful when you want to quickly generate a summary of a complete document without the need for complex processing."
    )
            summary_stuff_document_chain = stuff_document_chain_summarization(pdf_path)
            st.write(summary_stuff_document_chain)
            
            st.write("### 4. Map-Reduce Summarization from PDF")
            st.write("**Description:** Summarizes by dividing the text into chunks and then combining the summaries.")
            st.write(
        "- **When to Use:** Ideal for large documents where processing the entire text in one go is impractical due to size.\n"
        "- **Why:** This method breaks down the document into smaller parts, making it easier to handle large amounts of text and produce a summary that reflects the entire content."
    )    
            summary_map_reduce = map_reduce_summarization(pdf_path)
            st.write(summary_map_reduce)
            
            st.write("### 5. Map-Reduce Summarization with Custom Prompts from PDF")
            st.write("**Description:** Uses custom prompts for summarizing chunks and combining summaries.")
            st.write(
        "- **When to Use:** Useful when you need more control over the summarization process, such as incorporating specific instructions or focusing on key aspects.\n"
        "- **Why:** Custom prompts allow for tailored summarization that can better fit particular needs or stylistic requirements, offering greater flexibility compared to standard methods."
    ) 
            summary_map_reduce_custom = map_reduce_custom_prompts(pdf_path)
            st.write(summary_map_reduce_custom)
            
            st.write("### 6. RefineChain Summarization from PDF")
            st.write("**Description:** Summarizes the text using RefineChain, iteratively refining the summary.")
            st.write(
        "- **When to Use:** Best for documents requiring a high-quality, polished summary that captures nuanced details and provides a refined output.\n"
        "- **Why:** The iterative refinement process helps in producing a more coherent and polished summary, making it suitable for high-stakes content where clarity and detail are crucial."
    )
            summary_refine_chain = refine_chain_summarization(pdf_path)
            st.write(summary_refine_chain)

if __name__ == "__main__":
    main()

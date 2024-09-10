from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain.schema import SystemMessage, HumanMessage
from langchain.chains import LLMChain
from langchain.docstore.document import Document
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader
import os
from dotenv import load_dotenv

# Load .env file to access API keys
load_dotenv()

# Get OpenAI API key from the .env file
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize the OpenAI model with the API key
llm = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0, openai_api_key=OPENAI_API_KEY)

# Function 1: Basic Summarization from PDF
def basic_summarization_from_pdf(pdf_path):
    # Extract text from PDF
    pdf_reader = PdfReader(pdf_path)
    text = ''
    for page in pdf_reader.pages:
        content = page.extract_text()
        if content:
            text += content
    
    # Perform basic summarization
    chat_messages = [
        SystemMessage(content='You are an expert assistant with expertise in summarizing speeches'),
        HumanMessage(content=f'Please provide a short and concise summary of the following speech:\n TEXT: {text}')
    ]
    summary = llm(chat_messages).content
    # print("Basic Summarization from PDF:\n", summary)
    return summary


# Function 2: Prompt Template Summarization (Translated to Hindi) from PDF
def prompt_template_summarization_from_pdf(pdf_path, language):
    # Extract text from PDF
    pdf_reader = PdfReader(pdf_path)
    text = ''
    for page in pdf_reader.pages:
        content = page.extract_text()
        if content:
            text += content
    
    # Perform summarization with translation
    template = '''
    Write a summary of the following speech:
    Speech : `{speech}`
    Translate the precise summary to {language}.
    '''
    prompt = PromptTemplate(input_variables=['speech', 'language'], template=template)
    formatted_prompt = prompt.format(speech=text, language=language)
    llm_chain = LLMChain(llm=llm, prompt=prompt)
    summary = llm_chain.run({'speech': text, 'language': language})
    # print(f"\nPrompt Template Summarization (Translated to {language}) from PDF:\n", summary)
    return summary

# Function 3: StuffDocumentChain Summarization
def stuff_document_chain_summarization(pdf_path):
    # Extract text from PDF
    pdf_reader = PdfReader(pdf_path)
    text = ''
    for page in pdf_reader.pages:
        content = page.extract_text()
        if content:
            text += content
    
    # Summarize the document
    docs = [Document(page_content=text)]
    chain = load_summarize_chain(llm=llm, chain_type='stuff', verbose=False)
    summary = chain.run(docs)
    # print("\nStuffDocumentChain Summarization:\n", summary)
    return summary

# Function 4: Map-Reduce Summarization
def map_reduce_summarization(pdf_path):
    # Extract text from PDF
    pdf_reader = PdfReader(pdf_path)
    text = ''
    for page in pdf_reader.pages:
        content = page.extract_text()
        if content:
            text += content
    
    # Split the text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=20)
    chunks = text_splitter.create_documents([text])
    
    # Summarize using map-reduce
    chain = load_summarize_chain(llm=llm, chain_type='map_reduce', verbose=False)
    summary = chain.run(chunks)
    # print("\nMap-Reduce Summarization:\n", summary)
    return summary

# Function 5: Map-Reduce with Custom Prompts
def map_reduce_custom_prompts(pdf_path):
    # Extract text from PDF
    pdf_reader = PdfReader(pdf_path)
    text = ''
    for page in pdf_reader.pages:
        content = page.extract_text()
        if content:
            text += content
    
    # Split the text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    chunks = text_splitter.create_documents([text])
    
    # Custom prompts for map and final combination
    chunks_prompt = "Please summarize the below speech:\nSpeech:`{text}`\nSummary:"
    final_combine_prompt = '''
    Provide a final summary of the entire speech with these important points.
    Add a Generic Motivational Title,
    Start the precise summary with an introduction and provide the
    summary in numbered points for the speech.
    Speech: `{text}`
    '''
    
    map_prompt_template = PromptTemplate(input_variables=['text'], template=chunks_prompt)
    final_combine_prompt_template = PromptTemplate(input_variables=['text'], template=final_combine_prompt)
    
    summary_chain = load_summarize_chain(
        llm=llm,
        chain_type='map_reduce',
        map_prompt=map_prompt_template,
        combine_prompt=final_combine_prompt_template,
        verbose=False
    )
    summary = summary_chain.run(chunks)
    # print("\nMap-Reduce Summarization with Custom Prompts:\n", summary)
    return summary

# Function 6: RefineChain Summarization
def refine_chain_summarization(pdf_path):
    # Extract text from PDF
    pdf_reader = PdfReader(pdf_path)
    text = ''
    for page in pdf_reader.pages:
        content = page.extract_text()
        if content:
            text += content
    
    # Split the text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    chunks = text_splitter.create_documents([text])
    
    # Summarize using refine chain
    chain = load_summarize_chain(llm=llm, chain_type='refine', verbose=True)
    summary = chain.run(chunks)
    # print("\nRefineChain Summarization:\n", summary)
    return summary


# # Example usage:
# pdf_path = 'Machine_Learning_Engineer_Resume.pdf'

# # 1. Basic Summarization from PDF
# basic_summarization_from_pdf(pdf_path)

# # 2. Prompt Template Summarization (Translated to Hindi) from PDF
# prompt_template_summarization_from_pdf(pdf_path, language='English')

# # 3. StuffDocumentChain Summarization from PDF
# stuff_document_chain_summarization(pdf_path)

# # 4. Map-Reduce Summarization from PDF
# map_reduce_summarization(pdf_path)

# # 5. Map-Reduce Summarization with Custom Prompts from PDF
# map_reduce_custom_prompts(pdf_path)

# # 6. RefineChain Summarization from PDF
# refine_chain_summarization(pdf_path)

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
import os

def load_prompt():
    prompt = """You are an assistant that answers questions based on the provided PDF content.
    Given the following context and question, provide an accurate answer:
    Context: {context}
    Question: {question}
    If the answer is not in the provided context, respond with "I don't have enough information to answer that question."
    """
    return ChatPromptTemplate.from_template(prompt)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def create_rag_chain(vector_store):
    prompt = load_prompt()
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, api_key=os.getenv("OPENAI_API_KEY"))

    retriever = vector_store.as_retriever()
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain
import openai
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set the API key
openai.api_key = os.getenv("OPENAI_API_KEY")

def query_gpt(pdf_text, user_question):
    """Send the PDF text and the user question to OpenAI's GPT-3.5-turbo model."""
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an assistant that answers questions based on the provided PDF content."},
            {"role": "user", "content": f"PDF content: {pdf_text}"},
            {"role": "user", "content": user_question}
        ]
    )
    return response['choices'][0]['message']['content']

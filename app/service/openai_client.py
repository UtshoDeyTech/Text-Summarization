import os
import logging
import openai

logger = logging.getLogger(__name__)

# Set the API key
openai.api_key = os.getenv("OPENAI_API_KEY")

def get_embeddings(texts):
    try:
        if not isinstance(texts, list) or len(texts) == 0 or not all(isinstance(t, str) for t in texts):
            raise ValueError("Input must be a non-empty list of strings")
        
        logger.info(f"First text (truncated): {texts[0][:100]}...")
        
        # Check if we're using the new version of the openai library
        if hasattr(openai, 'Embedding'):
            response = openai.Embedding.create(
                input=texts,
                model="text-embedding-ada-002"
            )
            return [embedding.embedding for embedding in response.data]
        else:
            # Fallback for older versions of the openai library
            response = openai.Embedding.create(
                input=texts,
                model="text-embedding-ada-002"
            )
            return [embedding['embedding'] for embedding in response['data']]
    except Exception as e:
        logger.error(f"Error getting embeddings: {str(e)}")
        raise

# Test function
def test_get_embeddings():
    test_texts = ["Hello, world!", "This is a test."]
    try:
        embeddings = get_embeddings(test_texts)
        print(f"Successfully generated {len(embeddings)} embeddings.")
        print(f"First embedding (first 5 values): {embeddings[0][:5]}")
    except Exception as e:
        print(f"Error in test_get_embeddings: {str(e)}")

if __name__ == "__main__":
    test_get_embeddings()
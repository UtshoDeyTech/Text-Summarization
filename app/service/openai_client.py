import openai
from app.service.log_client import logger, log_error
from config import OPENAI_API_KEY

# Set the API key
openai.api_key = OPENAI_API_KEY

def get_embeddings(texts):
    try:
        if not isinstance(texts, list) or len(texts) == 0 or not all(isinstance(t, str) for t in texts):
            raise ValueError("Input must be a non-empty list of strings")
        
        # Log with text length and sample for better context
        logger.info(
            f"Getting embeddings | text_count={len(texts)}, "
            f"first_text_sample={texts[0][:100]}..., "
            f"total_characters={sum(len(t) for t in texts)}"
        )
        
        # Check if we're using the new version of the openai library
        if hasattr(openai, 'Embedding'):
            response = openai.Embedding.create(
                input=texts,
                model="text-embedding-ada-002"
            )
            embeddings = [embedding.embedding for embedding in response.data]
        else:
            # Fallback for older versions of the openai library
            response = openai.Embedding.create(
                input=texts,
                model="text-embedding-ada-002"
            )
            embeddings = [embedding['embedding'] for embedding in response['data']]
            
        logger.info(
            f"Successfully generated embeddings | "
            f"count={len(embeddings)}, "
            f"dimensions={len(embeddings[0])}"
        )
        
        return embeddings
        
    except Exception as e:
        error_msg = f"Error getting embeddings | error_type={type(e).__name__}, error={str(e)}"
        logger.error(error_msg)
        raise

# Test function
def test_get_embeddings():
    test_texts = ["Hello, world!", "This is a test."]
    try:
        logger.info(f"Starting embedding test | test_texts_count={len(test_texts)}")
        embeddings = get_embeddings(test_texts)
        logger.info(
            f"Test successful | embeddings_count={len(embeddings)}, "
            f"first_embedding_sample={embeddings[0][:5]}"
        )
    except Exception as e:
        logger.error(f"Test failed | error_type={type(e).__name__}, error={str(e)}")

if __name__ == "__main__":
    test_get_embeddings()
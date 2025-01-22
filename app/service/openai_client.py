from openai import AsyncOpenAI  # Change to AsyncOpenAI
from app.service.log_client import logger
from config import OPENAI_API_KEY

# Initialize the async client
client = AsyncOpenAI(api_key=OPENAI_API_KEY)  # Use AsyncOpenAI

async def get_embeddings(texts):
    """Get embeddings for a list of texts using OpenAI's API."""
    try:
        if not isinstance(texts, list) or len(texts) == 0 or not all(isinstance(t, str) for t in texts):
            raise ValueError("Input must be a non-empty list of strings")
        
        logger.info(
            f"Getting embeddings | text_count={len(texts)}, "
            f"first_text_sample={texts[0][:100]}..., "
            f"total_characters={sum(len(t) for t in texts)}"
        )
        
        response = await client.embeddings.create(  # Use await here
            input=texts,
            model="text-embedding-ada-002"
        )
        
        embeddings = [embedding.embedding for embedding in response.data]
        
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
async def test_embeddings():
    test_texts = ["Hello, world!", "This is a test."]
    try:
        logger.info(f"Starting embedding test | test_texts_count={len(test_texts)}")
        embeddings = await get_embeddings(test_texts)
        logger.info(
            f"Test successful | embeddings_count={len(embeddings)}, "
            f"first_embedding_sample={embeddings[0][:5]}"
        )
    except Exception as e:
        logger.error(f"Test failed | error_type={type(e).__name__}, error={str(e)}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_embeddings())
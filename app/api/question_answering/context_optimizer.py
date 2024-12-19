from typing import List
import tiktoken
from app.service.log_client import logger

class ContextOptimizer:
    def __init__(self, model: str = "gpt-3.5-turbo"):
        self.encoder = tiktoken.encoding_for_model(self._get_base_model(model))
        self.token_limits = {
            "gpt-3.5-turbo": 2000,    # Conservative limit for GPT-3.5
            "gpt-4": 4000,            # Conservative limit for GPT-4
            "gpt-4o": 4000,           # Same limit as GPT-4
            "gpt-4-32k": 20000        # Conservative limit for GPT-4-32k
        }
        self.max_tokens = self.token_limits.get(model, 2000)  # Default to 2000 if model not found
        
    def _get_base_model(self, model: str) -> str:
        """Convert custom model names to their base model for tiktoken."""
        model_mapping = {
            "gpt-4o": "gpt-4",
            "gpt-3.5-turbo": "gpt-3.5-turbo",
            "gpt-4": "gpt-4",
            "gpt-4-32k": "gpt-4-32k"
        }
        return model_mapping.get(model, "gpt-3.5-turbo")
        
    def count_tokens(self, text: str) -> int:
        try:
            return len(self.encoder.encode(text))
        except Exception as e:
            logger.error(f"Error counting tokens: {str(e)}")
            return len(text.split()) * 2
            
    def optimize_batch(self, contexts: List[dict], max_tokens: int) -> List[dict]:
        optimized = []
        current_tokens = 0
        
        # Add buffer for system messages and other overhead
        effective_max_tokens = max(100, max_tokens - 500)  # Leave 500 tokens buffer
        
        for ctx in contexts:
            tokens = self.count_tokens(ctx["text"])
            if current_tokens + tokens <= effective_max_tokens:
                optimized.append(ctx)
                current_tokens += tokens
            else:
                # If the context is too large, try to include a portion
                if tokens > 800:  # Reduced from 1000 for more conservative chunking
                    sentences = ctx["text"].split(". ")
                    current_chunk = []
                    chunk_tokens = 0
                    
                    for sentence in sentences:
                        sentence_tokens = self.count_tokens(sentence)
                        if chunk_tokens + sentence_tokens <= effective_max_tokens - current_tokens:
                            current_chunk.append(sentence)
                            chunk_tokens += sentence_tokens
                        else:
                            break
                            
                    if current_chunk:
                        optimized.append({
                            **ctx,
                            "text": ". ".join(current_chunk) + ".",
                            "is_truncated": True
                        })
                break  # Stop processing more contexts if we've hit the token limit
                
        return optimized

    def get_max_completion_tokens(self, model: str) -> int:
        """Get the maximum completion tokens for a given model."""
        completion_tokens = {
            "gpt-3.5-turbo": 300,
            "gpt-4": 400,
            "gpt-4o": 400,
            "gpt-4-32k": 1000
        }
        return completion_tokens.get(model, 300)
from typing import List
import tiktoken
from app.service.log_client import logger

class ContextOptimizer:
    def __init__(self, model: str = "gpt-3.5-turbo"):
        self.encoder = tiktoken.encoding_for_model(model)
        self.token_limits = {
            "gpt-3.5-turbo": 3000,
            "gpt-4": 6000,
            "gpt-4-32k": 28000
        }
        self.max_tokens = self.token_limits.get(model, 3000)
        
    def count_tokens(self, text: str) -> int:
        try:
            return len(self.encoder.encode(text))
        except Exception as e:
            logger.error(f"Error counting tokens: {str(e)}")
            return len(text.split()) * 2
            
    def optimize_batch(self, contexts: List[dict], max_tokens: int) -> List[dict]:
        optimized = []
        current_tokens = 0
        
        for ctx in contexts:
            tokens = self.count_tokens(ctx["text"])
            if current_tokens + tokens <= max_tokens:
                optimized.append(ctx)
                current_tokens += tokens
            else:
                if tokens > 1000:
                    sentences = ctx["text"].split(". ")
                    current_chunk = []
                    chunk_tokens = 0
                    
                    for sentence in sentences:
                        sentence_tokens = self.count_tokens(sentence)
                        if chunk_tokens + sentence_tokens <= max_tokens - current_tokens:
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
                
        return optimized
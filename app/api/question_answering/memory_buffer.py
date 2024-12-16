from typing import List, Tuple
from collections import deque
from datetime import datetime, timezone
import openai
from app.service.log_client import logger

class MemoryBuffer:
    def __init__(self, buffer_size=3):
        self.conversation_history = deque(maxlen=buffer_size)
        self.context_history = deque(maxlen=buffer_size)
        
    async def generate_context_summary(self, contexts: List[dict]) -> str:
        context_text = "\n".join([
            f"Source: {ctx['filename']} ({ctx['source_type']})\nContent: {ctx['text']}"
            for ctx in contexts
        ])
        
        response = await openai.ChatCompletion.acreate(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Create a brief summary of the key information from these documents."},
                {"role": "user", "content": context_text}
            ],
            temperature=0.7,
            max_tokens=150
        )
        return response.choices[0].message.content

    async def add_interaction(self, question: str, answer: str, contexts: List[dict]):
        summary = await self.generate_context_summary(contexts)
        self.conversation_history.append({
            "question": question,
            "answer": answer,
            "timestamp": datetime.now(timezone.utc),
            "context_summary": summary,
            "sources": contexts  # Store full context objects instead of just filename and type
        })
        self.context_history.append(contexts)

    def get_conversation_context(self):
        messages = []
        for item in self.conversation_history:
            messages.extend([
                {"role": "user", "content": item["question"]},
                {
                    "role": "assistant", 
                    "content": f"{item['answer']}\nContext: {item['context_summary']}",
                    "sources": item['sources']  # Include full source information
                }
            ])
        return messages

    def is_followup_question(self, current_question: str) -> Tuple[bool, List[dict]]:
        if not self.conversation_history:
            return False, []
            
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Determine if the new question is a follow-up to the previous conversation."},
                {"role": "user", "content": f"Previous conversation:\n{self.conversation_history[-1]['question']}\n{self.conversation_history[-1]['answer']}\n\nNew question: {current_question}"}
            ],
            temperature=0.3,
            max_tokens=50
        )
        
        is_followup = "yes" in response.choices[0].message.content.lower()
        relevant_contexts = self.context_history[-1] if is_followup else []
        
        return is_followup, relevant_contexts
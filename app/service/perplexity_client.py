from config import PERPLEXITY_API_KEY
import requests
from typing import Optional

class PerplexityClient:
    def __init__(self, api_key: str = PERPLEXITY_API_KEY):
        self.api_key = api_key
        self.base_url = "https://api.perplexity.ai/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
    
    async def ask_question(self, question: str, model: str = "sonar-pro") -> str:
        """
        Ask a question to Perplexity AI and get the answer
        
        Args:
            question: The question to ask
            model: The model to use (default: "sonar-pro")
            
        Returns:
            The answer from Perplexity AI
            
        Raises:
            Exception: If the API request fails
        """
        
        # Hardcoded system prompt for real estate queries
        system_prompt = """You are a real estate expert assistant. Your task is to help users find property information including prices, details, and values for specific addresses.

Instructions:
1. Always provide the most accurate and up-to-date property information available
2. If you cannot find exact information for the specific address, provide the closest available information (nearby properties, area averages, etc.)
3. For property prices, include recent sale prices, current market estimates, and rental prices if available
4. For property details, include: square footage, bedrooms, bathrooms, lot size, year built, property type, etc.
5. If dealing with commercial properties (gas stations, corporate buildings), focus on commercial real estate data

CRITICAL: You MUST respond ONLY with a valid JSON object in exactly this format:
{
    "answer": "Your detailed property information response here",
    "source": [
        "Source 1 name and URL if available",
        "Source 2 name and URL if available",
        "Source 3 name and URL if available"
    ]
}

REQUIREMENTS:
- Return ONLY the JSON object, no other text before or after
- The answer field should contain all the property information in a clear, readable format
- The source field must be an array with at least 2-3 reliable sources with names and URLs
- Ensure the JSON is properly formatted and valid
- Do not include markdown formatting, just pure JSON"""
        data = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question}
            ]
        }
        
        response = requests.post(self.base_url, json=data, headers=self.headers)
        
        if response.status_code == 200:
            result = response.json()
            answer = result['choices'][0]['message']['content']
            return answer
        else:
            raise Exception(f"Perplexity API request failed with status code {response.status_code}: {response.text}")

# Create a singleton instance
perplexity_client = PerplexityClient()
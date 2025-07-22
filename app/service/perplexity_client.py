from config import PERPLEXITY_API_KEY
import requests
import json
import re
from datetime import datetime
from typing import Dict, Any, List
from collections import defaultdict, deque
from bs4 import BeautifulSoup

class PerplexityClient:
    def __init__(self, api_key: str = PERPLEXITY_API_KEY):
        self.api_key = api_key
        self.base_url = "https://api.perplexity.ai/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        # Store conversation history for each user (max 3 Q&A pairs)
        self.conversation_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=3))
    
    def ask_question(self, question: str, user_id: str = None, model: str = "sonar-pro") -> Dict[str, Any]:
        """
        Ask Perplexity AI for property information in HTML table format
        
        Args:
            question (str): The question to ask
            user_id (str): Optional user ID to maintain conversation history
            model (str): The model to use (default: "sonar-pro")
            
        Returns:
            Dict[str, Any]: Dictionary containing HTML table, sources, suggested questions, and conversation context
        """
        
        # Smart system prompt that adapts based on question type and context
        system_prompt = """You are a real estate expert. Always respond with comprehensive property information in this EXACT detailed HTML table format:

<table><thead><tr><th>Category</th><th>Details</th></tr></thead><tbody>
[Create detailed breakdown with category headers and multiple sub-rows for each category]
</tbody></table>

DETAILED TABLE STRUCTURE - Use this pattern:
- Category header row: <tr><td colspan="2"><strong>CATEGORY NAME</strong></td></tr>
- Multiple sub-rows: <tr><td>Sub-Detail Name</td><td>Detailed information with citations [1][2]</td></tr>

REQUIRED CATEGORIES (choose relevant ones based on question):
• Property Details: Type, Address, Size, Year Built, Condition, Ownership, Environmental
• Property Features: Location, Traffic, Income Streams, Financial Performance, Competition
• Market Information: Valuation Methods, Industry Multiples, Market Trends, Demand
• Comparative Analysis: (only if comparing properties) Side-by-side comparison details
• Investment Analysis: ROI, Risks, Growth Potential, Market Position
• Location Info: Area demographics, Development, Transportation, Future Outlook

SMART RESPONSE LOGIC:
- COMPARISON QUESTIONS: If user asks "compare", "vs", "difference between", or references previous properties, create "Comparative Analysis" section comparing current property with conversation history
- VALUATION QUESTIONS: Focus on "Market Information" and "Investment Analysis" sections
- GENERAL PROPERTY QUESTIONS: Use "Property Details", "Property Features", "Location Info"
- FOLLOW-UP QUESTIONS: Reference previous conversation context intelligently

CONTENT REQUIREMENTS:
- Break down each category into 5-8 specific sub-details
- Provide comprehensive information for each sub-detail
- Always cite sources [1][2][3] within content
- Use conversation history context for comparisons when relevant
- If specific data unavailable, provide typical/comparable information for that property type/area

COMPARISON HANDLING:
- When user asks for comparison, analyze conversation history
- Create detailed side-by-side comparison in "Comparative Analysis" section
- Include property A vs property B details, key differences, investment implications
- Reference specific details from previous queries in your response

Keep table structure raw HTML without any CSS styling or style attributes. Use only <strong> tags for category headers."""

        # Build messages with conversation history
        messages = [{"role": "system", "content": system_prompt}]
        
        # Add conversation history if available
        if user_id and user_id in self.conversation_history:
            for qa_pair in self.conversation_history[user_id]:
                messages.append({"role": "user", "content": qa_pair["question"]})
                messages.append({"role": "assistant", "content": qa_pair["answer_text"]})
        
        # Add current question
        messages.append({"role": "user", "content": question})

        data = {
            "model": model,
            "messages": messages,
            "return_citations": True,
            "return_images": False,
            "return_related_questions": False  # We generate our own
        }
        
        try:
            response = requests.post(
                url=self.base_url, 
                json=data, 
                headers=self.headers, 
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                content = self._extract_content(result)
                
                if not content:
                    return self._create_error_response("No content in API response")
                
                # Extract/create HTML table
                html_table = self._extract_or_create_table(content)
                
                # Extract text from table for conversation history (cost optimization)
                answer_text = self._extract_text_from_table(html_table)
                
                # Extract sources
                sources = self._extract_sources(content, result.get("citations", []), result)
                
                # Generate suggested questions with user context
                suggested_questions = self._generate_suggested_questions(question, user_id)
                
                # Store conversation history with text only (not HTML)
                if user_id:
                    self._store_conversation(user_id, question, answer_text)
                
                return {
                    "html_table": html_table,
                    "sources": sources,
                    "suggested_questions": suggested_questions,
                    "summary": "Property information table",
                    "raw_response": content,
                    "conversation_context": len(self.conversation_history.get(user_id, [])),
                    "debug_info": {  # Add debug info to help troubleshoot
                        "has_citations_param": bool(result.get("citations")),
                        "citations_count": len(result.get("citations", [])),
                        "has_web_results": "web_results" in result,
                        "content_has_citation_numbers": bool(re.findall(r'\[(\d+)\]', content)),
                        "api_response_keys": list(result.keys())
                    }
                }
            else:
                return self._create_error_response(f"API error {response.status_code}: {response.text}")
                
        except requests.Timeout:
            return self._create_error_response("Request timeout")
        except requests.RequestException as e:
            return self._create_error_response(f"Request error: {str(e)}")
        except Exception as e:
            return self._create_error_response(f"Unexpected error: {str(e)}")
    
    def _extract_content(self, result: Dict) -> str:
        """Extract content from API response"""
        if "choices" in result and len(result["choices"]) > 0:
            choice = result["choices"][0]
            if "message" in choice and "content" in choice["message"]:
                return choice["message"]["content"]
        return ""
    
    def _extract_text_from_table(self, html_table: str) -> str:
        """Extract plain text from detailed breakdown table for conversation history"""
        try:
            soup = BeautifulSoup(html_table, 'html.parser')
            
            # Extract key information from detailed table
            text_parts = []
            current_category = ""
            
            for row in soup.find_all('tr'):
                cells = row.find_all(['td', 'th'])
                if len(cells) == 2:
                    # Skip header row
                    if cells[0].name == 'th':
                        continue
                    
                    # Check if this is a category header row
                    if cells[0].get('colspan') == '2':
                        current_category = cells[0].text.strip()
                        continue
                    
                    # Regular detail row
                    detail = cells[0].text.strip()
                    value = cells[1].text.strip()
                    
                    if detail and value:
                        # Keep it concise for conversation history
                        short_value = value[:80] + "..." if len(value) > 80 else value
                        text_parts.append(f"{detail}: {short_value}")
            
            # Limit total length to save API costs
            full_text = "; ".join(text_parts)
            return full_text[:300] + "..." if len(full_text) > 300 else full_text
            
        except Exception:
            # Fallback: simple text extraction
            clean_text = re.sub(r'<[^>]+>', '', html_table)
            clean_text = re.sub(r'\s+', ' ', clean_text).strip()
            return clean_text[:200] + "..." if len(clean_text) > 200 else clean_text
    
    def _store_conversation(self, user_id: str, question: str, answer_text: str):
        """Store conversation history with text only (not HTML)"""
        if not user_id:
            return
        
        qa_pair = {
            "question": question,
            "answer_text": answer_text,
            "timestamp": datetime.now().isoformat()
        }
        
        self.conversation_history[user_id].append(qa_pair)
    
    def _extract_or_create_table(self, content: str) -> str:
        """Extract existing table or create one from content"""
        content = content.replace('\\n', '').replace('\n', ' ').strip()
        
        # Look for existing table
        if '<table>' in content and '</table>' in content:
            start = content.find('<table>')
            end = content.find('</table>') + 8
            table = content[start:end]
            return self._clean_table(table)
        
        # Create table from content
        return self._create_table_from_text(content)
    
    def _clean_table(self, html_table: str) -> str:
        """Clean up table formatting"""
        # Remove escaped quotes and extra spaces
        cleaned = html_table.replace('\\"', '"').replace("\\'", "'")
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        return cleaned
    
    def _create_table_from_text(self, text: str) -> str:
        """Create detailed breakdown HTML table from text content"""
        # Extract basic property info from text
        info = self._parse_property_info_detailed(text)
        
        html = '<table><thead><tr><th>Category</th><th>Details</th></tr></thead><tbody>'
        
        if info:
            for category, items in info.items():
                if items:
                    # Add category header row with no CSS
                    html += f'<tr><td colspan="2"><strong>{category}</strong></td></tr>'
                    # Add detail rows
                    for detail, value in items:
                        html += f'<tr><td>{detail}</td><td>{value}</td></tr>'
        else:
            html += '<tr><td colspan="2"><strong>Property Information</strong></td></tr>'
            html += '<tr><td>Status</td><td>Property details available upon research</td></tr>'
        
        html += '</tbody></table>'
        return html
    
    def _parse_property_info_detailed(self, text: str) -> Dict[str, List[tuple]]:
        """Parse property information for detailed breakdown format"""
        info = {
            "Property Details": [],
            "Property Features": [],
            "Market Information": []
        }
        
        # Extract address
        address_match = re.search(r'(\d+\s+[^,\n]+(?:,\s*[^,\n]+){1,3})', text)
        if address_match:
            info["Property Details"].append(("Address", address_match.group(1)))
        
        # Extract property type
        if 'gas station' in text.lower():
            info["Property Details"].append(("Property Type", "Gas Station & Convenience Store"))
        elif 'single.family' in text.lower() or 'single family' in text.lower():
            info["Property Details"].append(("Property Type", "Single Family Residence"))
        elif 'commercial' in text.lower():
            info["Property Details"].append(("Property Type", "Commercial Property"))
        elif 'condo' in text.lower():
            info["Property Details"].append(("Property Type", "Condominium"))
        
        # Add placeholder details based on content
        if 'gas station' in text.lower():
            info["Property Details"].extend([
                ("Ownership Structure", "Owned or leased property impacts valuation significantly"),
                ("Size/Features", "Site size, number of dispensers, and store square footage are key factors"),
                ("Environmental Compliance", "Underground storage tanks and environmental regulations affect value")
            ])
            
            info["Property Features"].extend([
                ("Location", "High-traffic location and highway access increase value"),
                ("Traffic Patterns", "Visibility and traffic volume on adjacent roads are critical"),
                ("Income Streams", "Fuel sales, convenience store, and additional services enhance value"),
                ("Competition", "Density of nearby gas stations affects market share")
            ])
            
            info["Market Information"].extend([
                ("Valuation Method", "Income approach, sales comparison, and cost approach commonly used"),
                ("Industry Multiples", "Gas stations typically command higher multiples than convenience stores alone"),
                ("Market Demand", "Local fuel demand and population growth drive valuation")
            ])
        
        # Extract prices
        prices = re.findall(r'\$[\d,]+', text)
        if prices:
            info["Market Information"].append(("Current Value", prices[0]))
        
        # Extract year
        year_match = re.search(r'\b(19|20)\d{2}\b', text)
        if year_match:
            info["Property Details"].append(("Year Built", year_match.group(0)))
        
        # Extract square footage
        sqft_match = re.search(r'(\d+[\d,]*)\s*sq\.?\s*ft\.?', text, re.IGNORECASE)
        if sqft_match:
            info["Property Details"].append(("Square Footage", f"{sqft_match.group(1)} sq ft"))
        
        return {k: v for k, v in info.items() if v}  # Remove empty categories
    
    def _extract_sources(self, content: str, citations: List, full_response: Dict) -> List[str]:
        """Extract sources from response with comprehensive fallback"""
        sources = []
        
        # Method 1: From citations parameter
        if citations:
            for citation in citations:
                if isinstance(citation, dict):
                    if "url" in citation:
                        title = citation.get("title", citation.get("name", ""))
                        url = citation["url"]
                        sources.append(f"{title}: {url}" if title else self._format_url(url))
                    elif "source" in citation:
                        sources.append(str(citation["source"]))
                elif isinstance(citation, str):
                    sources.append(citation)
        
        # Method 2: From full_response.citations
        if "citations" in full_response:
            response_citations = full_response["citations"]
            for citation in response_citations:
                if isinstance(citation, dict) and "url" in citation:
                    title = citation.get("title", citation.get("name", ""))
                    url = citation["url"]
                    source_text = f"{title}: {url}" if title else self._format_url(url)
                    if source_text not in sources:
                        sources.append(source_text)
        
        # Method 3: From web_results in response
        if "web_results" in full_response:
            for result in full_response["web_results"]:
                if isinstance(result, dict) and "url" in result:
                    title = result.get("name", result.get("title", ""))
                    url = result["url"]
                    source_text = f"{title}: {url}" if title else self._format_url(url)
                    if source_text not in sources:
                        sources.append(source_text)
        
        # Method 4: From URLs in content
        url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+(?:[^\s.,;!?<>"{}|\\^`\[\]])'
        urls = re.findall(url_pattern, content)
        for url in urls:
            # Clean trailing punctuation
            url = re.sub(r'[.,;!?\]]+$', '', url)
            source_text = self._format_url(url)
            if source_text not in sources:
                sources.append(source_text)
        
        # Method 5: Look for citation numbers in content [1], [2], etc.
        citation_numbers = re.findall(r'\[(\d+)\]', content)
        if citation_numbers and not sources:
            # If we found citation numbers but no actual sources, add generic ones
            sources.append("Sources: Information compiled from real estate databases and industry reports")
        
        # Method 6: Check for nested sources in API response
        if "choices" in full_response:
            for choice in full_response["choices"]:
                if isinstance(choice, dict):
                    if "citations" in choice:
                        for citation in choice["citations"]:
                            if isinstance(citation, dict) and "url" in citation:
                                title = citation.get("title", citation.get("name", ""))
                                url = citation["url"]
                                source_text = f"{title}: {url}" if title else self._format_url(url)
                                if source_text not in sources:
                                    sources.append(source_text)
                    elif "message" in choice and "citations" in choice["message"]:
                        for citation in choice["message"]["citations"]:
                            if isinstance(citation, dict) and "url" in citation:
                                title = citation.get("title", citation.get("name", ""))
                                url = citation["url"]
                                source_text = f"{title}: {url}" if title else self._format_url(url)
                                if source_text not in sources:
                                    sources.append(source_text)
        
        # Remove duplicates while preserving order
        unique_sources = []
        seen = set()
        for source in sources:
            if source not in seen:
                seen.add(source)
                unique_sources.append(source)
        
        # Fallback if no sources found
        if not unique_sources:
            unique_sources = ["Sources: Real estate databases and industry reports"]
        
        return unique_sources
    
    def _format_url(self, url: str) -> str:
        """Format URL with site name"""
        site_map = {
            'realtor.com': 'Realtor.com',
            'zillow.com': 'Zillow',
            'trulia.com': 'Trulia',
            'redfin.com': 'Redfin',
            'loopnet.com': 'LoopNet',
            'apartments.com': 'Apartments.com',
            'propertyrecords.com': 'PropertyRecords.com',
            'cityfeet.com': 'CityFeet',
            'myelisting.com': 'MyeListing',
            '7-eleven.com': '7-Eleven',
            'homes.com': 'Homes.com',
            'propertyshark.com': 'PropertyShark',
            'crexi.com': 'Crexi'
        }
        
        for domain, name in site_map.items():
            if domain in url.lower():
                return f"{name}: {url}"
        
        # For unknown domains
        domain_match = re.search(r'://(?:www\.)?([^/]+)', url)
        if domain_match:
            domain_name = domain_match.group(1).split('.')[0].capitalize()
            return f"{domain_name}: {url}"
        
        return f"Source: {url}"
    
    def _generate_suggested_questions(self, original_question: str, user_id: str = None) -> List[str]:
        """Generate smart suggested follow-up questions based on context"""
        question_lower = original_question.lower()
        
        # Check if user has conversation history for context-aware suggestions
        has_history = user_id and user_id in self.conversation_history and len(self.conversation_history[user_id]) > 0
        
        # Comparison questions
        if "compare" in question_lower or "vs" in question_lower or "difference" in question_lower:
            return [
                "What are the key investment advantages of each property?",
                "Which property offers better long-term appreciation potential?",
                "How do the operating costs compare between these properties?"
            ]
        
        # Valuation questions
        elif "value" in question_lower or "price" in question_lower or "valuation" in question_lower:
            if has_history:
                return [
                    "How does this valuation compare to the previous property?",
                    "What are the key differences in investment potential?",
                    "Which property would provide better ROI?"
                ]
            return [
                "What recent comparable sales support this valuation?",
                "How do current market trends affect the property value?",
                "What factors could increase or decrease future value?"
            ]
        
        # Commercial/Business property questions
        elif "commercial" in question_lower or "business" in question_lower or "gas station" in question_lower:
            if has_history:
                return [
                    "Compare the traffic patterns and visibility of both locations",
                    "Which property has better competitive positioning?",
                    "What are the operating expense differences?"
                ]
            return [
                "What is the traffic count and visibility analysis for this location?",
                "How does local competition affect business potential?",
                "What are typical operating expenses and profit margins?"
            ]
        
        # Investment questions
        elif "investment" in question_lower or "roi" in question_lower or "return" in question_lower:
            if has_history:
                return [
                    "Compare the investment risks between these properties",
                    "Which location offers better growth potential?",
                    "How do the cash flow projections differ?"
                ]
            return [
                "What is the expected return on investment for this property?",
                "What are the main risks and mitigation strategies?",
                "How does this compare to other investment opportunities?"
            ]
        
        # General property questions
        else:
            if has_history:
                return [
                    "How does this property compare to the previous one we discussed?",
                    "What are the key differences between these properties?",
                    "Which property would be a better choice for my needs?"
                ]
            return [
                "What are the comparable properties in this area?",
                "What is the neighborhood market trend and growth outlook?",
                "What are the property tax rates and local amenities?"
            ]
    
    def _create_error_response(self, error_msg: str) -> Dict[str, Any]:
        """Create error response with no CSS"""
        error_table = f"""<table><thead><tr><th>Category</th><th>Details</th></tr></thead><tbody>
<tr><td colspan="2"><strong>Error</strong></td></tr>
<tr><td>Message</td><td>{error_msg}</td></tr>
<tr><td>Action</td><td>Please try again with a different query or check the property address format</td></tr>
</tbody></table>"""
        
        return {
            "html_table": error_table,
            "sources": [],
            "suggested_questions": [
                "What property information is available?",
                "How can I find accurate property details?",
                "What are the best real estate sources?"
            ],
            "summary": f"Error: {error_msg}",
            "raw_response": error_msg,
            "conversation_context": 0
        }
    
    def get_conversation_history(self, user_id: str) -> List[Dict[str, Any]]:
        """Get conversation history for a user"""
        return list(self.conversation_history.get(user_id, []))
    
    def clear_conversation_history(self, user_id: str) -> bool:
        """Clear conversation history for a user"""
        if user_id in self.conversation_history:
            self.conversation_history[user_id].clear()
            return True
        return False

# Create client instance
perplexity_client = PerplexityClient()
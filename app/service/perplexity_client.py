from config import PERPLEXITY_API_KEY
import requests
import json
import re
import time
from datetime import datetime
from typing import Dict, Any, List
from collections import defaultdict, deque

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
        SYNCHRONOUS function to ask Perplexity AI for property information in HTML table format
        
        Args:
            question (str): The question to ask
            user_id (str): Optional user ID to maintain conversation history
            model (str): The model to use (default: "sonar-pro")
            
        Returns:
            Dict[str, Any]: Dictionary containing HTML table, sources, suggested questions, and conversation context
        """
        
        # Modified system prompt to ensure proper HTML formatting
        system_prompt = """You are a real estate expert assistant. Research comprehensive property information and present it in an organized 3-column HTML table format.

CRITICAL: You MUST respond with ONLY a single HTML table using this EXACT structure with proper spacing:

<table> <thead> <tr><th>Category</th><th>Detail</th><th>Value</th></tr> </thead> <tbody> [Your content here organized by categories] </tbody> </table>

FORMATTING REQUIREMENTS:
- Use 3 columns: Category, Detail, Value
- Group related information using rowspan for the Category column
- Add spaces between HTML tags for readability: <table> <thead> etc.
- Use regular quotes (") not escaped quotes (\")
- Keep content clean and well-formatted

CONTENT INSTRUCTIONS:
- If user asks for comparisons between properties from previous conversation, create a comparison table
- For comparisons, use categories like "Property A Details", "Property B Details", "Comparison Analysis"
- Choose the most relevant and available information for this specific property or comparison
- Focus on what's actually important and available for this property type and location

COMPARISON HANDLING:
- If user mentions "compare", "vs", "difference between", or refers to previous questions/properties, create a comparison table
- Use conversation history to identify which properties to compare
- Structure comparison with side-by-side details and analysis
- Include a "Key Differences" or "Comparison Summary" category

SUGGESTED CATEGORIES (use what makes sense):
- Property Details: Basic property information that's essential
- Property Features: Physical characteristics and amenities that matter
- Additional Information: Context, neighborhood, market info, or other relevant details
- Comparison Analysis: (for comparison queries) Key differences and insights

REQUIREMENTS:
- Research thoroughly using multiple real estate sources
- Focus on the EXACT address provided by the user
- Include citations in your research but format them as [1], [2], etc.
- Provide accurate, specific information when available
- Skip irrelevant fields (e.g., don't ask for bedrooms on a gas station)
- Consider previous conversation context if provided - USE IT INTELLIGENTLY
- Adapt your response to the property type (residential, commercial, etc.)
- Quality over quantity - include what's useful and available
- ALWAYS maintain tabular format even for comparisons, analysis, or follow-up questions

Let the research and conversation context guide what information you provide."""

        # Build messages array with conversation history
        messages = [{"role": "system", "content": system_prompt}]
        
        # Add conversation history if user_id is provided
        if user_id and user_id in self.conversation_history:
            history = self.conversation_history[user_id]
            for qa_pair in history:
                messages.append({"role": "user", "content": qa_pair["question"]})
                messages.append({"role": "assistant", "content": qa_pair["answer_summary"]})
        
        # Add current question
        messages.append({"role": "user", "content": question})

        data = {
            "model": model,
            "messages": messages,
            # Enable citations in the response
            "return_citations": True,
            "return_images": False
        }
        
        try:
            # Make synchronous HTTP request
            response = requests.post(
                url=self.base_url, 
                json=data, 
                headers=self.headers, 
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                
                # Extract content and citations
                content = None
                citations = []
                
                if "choices" in result and len(result["choices"]) > 0:
                    choice = result["choices"][0]
                    if "message" in choice and "content" in choice["message"]:
                        content = choice["message"]["content"]
                
                # Extract citations if available in the response
                if "citations" in result:
                    citations = result["citations"]
                elif "choices" in result and len(result["choices"]) > 0:
                    choice = result["choices"][0]
                    if "citations" in choice:
                        citations = choice["citations"]
                    elif "message" in choice and "citations" in choice["message"]:
                        citations = choice["message"]["citations"]
                
                if not content:
                    return self._create_error_response("No content in API response")
                
                # Process content to extract table
                html_table = self._extract_or_create_table(content)
                
                # Extract sources from multiple methods
                sources = self._extract_sources_comprehensive(content, citations, result)
                
                # Generate suggested questions based on the original question and history
                suggested_questions = self._generate_suggested_questions(question, user_id)
                
                # Store conversation history if user_id is provided
                if user_id:
                    self._store_conversation(user_id, question, html_table, content)
                
                response_data = {
                    "html_table": html_table,
                    "sources": sources,
                    "suggested_questions": suggested_questions,
                    "summary": "Property information table",
                    "raw_response": content,
                    "full_api_response": result,  # For debugging
                    "conversation_context": len(self.conversation_history.get(user_id, [])) if user_id else 0
                }
                
                return response_data
            else:
                return self._create_error_response(f"API error {response.status_code}: {response.text}")
                
        except requests.Timeout:
            return self._create_error_response("Request timeout")
        except requests.RequestException as e:
            return self._create_error_response(f"Request error: {str(e)}")
        except Exception as e:
            return self._create_error_response(f"Unexpected error: {str(e)}")
    
    def _store_conversation(self, user_id: str, question: str, html_table: str, raw_response: str):
        """Store conversation in history (max 3 Q&A pairs per user)"""
        if not user_id:
            return
            
        # Create a summary of the answer for context (shorter than full HTML table)
        answer_summary = self._create_answer_summary(html_table, raw_response)
        
        qa_pair = {
            "question": question,
            "answer_summary": answer_summary,
            "timestamp": datetime.now().isoformat()
        }
        
        # Add to deque (automatically removes oldest when exceeding maxlen=3)
        self.conversation_history[user_id].append(qa_pair)
    
    def _create_answer_summary(self, html_table: str, raw_response: str) -> str:
        """Create a concise summary of the answer for conversation context"""
        # Extract key information from the HTML table for context
        summary_parts = []
        
        # Try to extract address
        if "Address" in html_table:
            address_match = re.search(r'<td>Address</td><td>([^<]+)</td>', html_table)
            if address_match:
                summary_parts.append(f"Address: {address_match.group(1)}")
        
        # Try to extract property type
        if "Property Type" in html_table:
            type_match = re.search(r'<td>Property Type</td><td>([^<]+)</td>', html_table)
            if type_match:
                summary_parts.append(f"Type: {type_match.group(1)}")
        
        # Try to extract current market value
        if "Current Market Value" in html_table:
            value_match = re.search(r'<td>Current Market Value</td><td>([^<]+)</td>', html_table)
            if value_match:
                summary_parts.append(f"Value: {value_match.group(1)}")
        
        # If we got key info, return it
        if summary_parts:
            return "Property information provided: " + ", ".join(summary_parts)
        
        # Fallback to generic summary
        return "Property information table provided with available details"
    
    def get_conversation_history(self, user_id: str) -> List[Dict[str, Any]]:
        """Get conversation history for a user"""
        if user_id not in self.conversation_history:
            return []
        return list(self.conversation_history[user_id])
    
    def clear_conversation_history(self, user_id: str) -> bool:
        """Clear conversation history for a user"""
        if user_id in self.conversation_history:
            self.conversation_history[user_id].clear()
            return True
        return False
    
    def _generate_suggested_questions(self, original_question: str, user_id: str = None, is_comparison: bool = False) -> List[str]:
        """
        Generate 3 related suggested questions based on the original question and conversation history
        """
        try:
            # Build context from conversation history
            context_info = ""
            if user_id and user_id in self.conversation_history:
                history = list(self.conversation_history[user_id])
                if history:
                    context_info = f"\n\nPrevious conversation context:\n"
                    for i, qa_pair in enumerate(history[-2:], 1):  # Last 2 Q&As for context
                        context_info += f"Q{i}: {qa_pair['question']}\nA{i}: {qa_pair['answer_summary']}\n"
            
            # Adjust prompt based on whether this was a comparison
            if is_comparison:
                suggestion_prompt = f"""Based on this comparison question: "{original_question}"{context_info}

Generate exactly 3 related follow-up questions that would be logical next steps after a comparison. 
Focus on actionable insights, deeper analysis, or related property research questions.

Format your response as a simple numbered list:
1. [Question 1]
2. [Question 2] 
3. [Question 3]

Keep each question concise and actionable."""
            else:
                suggestion_prompt = f"""Based on this real estate question: "{original_question}"{context_info}

Generate exactly 3 related follow-up questions that a user might want to ask next. 
The questions should be practical, related to real estate research, and consider the conversation context if provided.
If there's previous context, suggest smart comparison or follow-up questions.

Format your response as a simple numbered list:
1. [Question 1]
2. [Question 2] 
3. [Question 3]

Keep each question concise and actionable."""

            data = {
                "model": "sonar-pro",
                "messages": [
                    {"role": "user", "content": suggestion_prompt}
                ],
                "return_citations": False,
                "return_images": False
            }
            
            response = requests.post(
                url=self.base_url,
                json=data,
                headers=self.headers,
                timeout=15  # Shorter timeout for suggestions
            )
            
            if response.status_code == 200:
                result = response.json()
                if "choices" in result and len(result["choices"]) > 0:
                    choice = result["choices"][0]
                    if "message" in choice and "content" in choice["message"]:
                        content = choice["message"]["content"]
                        return self._parse_suggested_questions(content)
            
            # Fallback if API call fails
            return self._generate_fallback_questions(original_question, user_id, is_comparison)
            
        except Exception:
            # Fallback if anything goes wrong
            return self._generate_fallback_questions(original_question, user_id, is_comparison)
    
    def _parse_suggested_questions(self, content: str) -> List[str]:
        """Parse suggested questions from AI response"""
        questions = []
        
        # Look for numbered list format
        lines = content.strip().split('\n')
        for line in lines:
            line = line.strip()
            # Match patterns like "1. Question" or "1) Question" or "• Question"
            match = re.match(r'^\d+[\.\)]\s*(.+)$', line)
            if match:
                questions.append(match.group(1).strip())
            elif line.startswith('•') or line.startswith('-'):
                # Handle bullet points
                question = line[1:].strip()
                if question:
                    questions.append(question)
        
        # If we couldn't parse properly, try to split by common patterns
        if not questions and content:
            # Try splitting by newlines and filtering
            potential_questions = [q.strip() for q in content.split('\n') if q.strip()]
            for pq in potential_questions:
                # Clean up common prefixes
                cleaned = re.sub(r'^\d+[\.\)]\s*', '', pq)
                cleaned = re.sub(r'^[•\-]\s*', '', cleaned)
                if cleaned and len(cleaned) > 10:  # Reasonable question length
                    questions.append(cleaned)
        
        # Ensure we have exactly 3 questions, pad or trim as needed
        if len(questions) > 3:
            questions = questions[:3]
        elif len(questions) < 3:
            # Pad with fallback questions
            fallback = self._generate_fallback_questions("", None)
            questions.extend(fallback[:3-len(questions)])
        
        return questions[:3]
    
    def _generate_fallback_questions(self, original_question: str, user_id: str = None, is_comparison: bool = False) -> List[str]:
        """Generate fallback questions when AI suggestions fail"""
        
        # Check conversation history for better context
        has_history = user_id and user_id in self.conversation_history and len(self.conversation_history[user_id]) > 0
        
        if is_comparison:
            # Comparison-specific fallback questions
            return [
                "What are the key investment advantages of each property?",
                "How do the locations compare in terms of foot traffic and visibility?",
                "Which property offers better long-term appreciation potential?"
            ]
        
        if has_history:
            # If user has history, suggest more advanced follow-ups or comparisons
            return [
                "How does this compare to the previous property we discussed?",
                "What are the key differences between these properties?",
                "Which property would be a better investment choice?"
            ]
        
        # Analyze the original question to provide relevant fallbacks
        question_lower = original_question.lower()
        
        if "valuation" in question_lower or "value" in question_lower:
            return [
                "What recent comparable sales support this valuation?",
                "How do market conditions affect the current valuation?",
                "What factors could increase or decrease the property value?"
            ]
        elif "gas station" in question_lower or "commercial" in question_lower:
            return [
                "What is the traffic count and location analysis for this property?",
                "How does the competition affect the business potential?",
                "What are the typical operating expenses and profit margins?"
            ]
        elif "property" in question_lower or "address" in question_lower:
            return [
                "What are the property tax rates in this area?",
                "What are comparable properties selling for nearby?",
                "What is the crime rate and school rating for this neighborhood?"
            ]
        elif "investment" in question_lower or "rental" in question_lower:
            return [
                "What is the expected rental yield for this property?",
                "What are the vacancy rates in this area?",
                "What maintenance and management costs should I expect?"
            ]
        else:
            # Generic real estate questions
            return [
                "What is the neighborhood market trend and price history?",
                "What are the local amenities and transportation options?",
                "What inspection issues or repairs might this property need?"
            ]
    
    def _extract_or_create_table(self, content: str) -> str:
        """Extract existing table or convert content to 3-column table format"""
        if not isinstance(content, str):
            content = str(content)
        
        # Clean up any \n characters first
        content = content.replace('\\n', '').replace('\n', ' ').strip()
        
        # Check if we got a 3-column table with Category, Detail, Value columns
        if '<th>Category</th><th>Detail</th><th>Value</th>' in content:
            return self._clean_table_formatting(content)
        elif content.strip().startswith('<table'):
            # Extract existing table and convert to 3-column format if needed
            table_end = content.find('</table>') + 8
            if table_end > 7:
                existing_table = content[:table_end].strip()
                return self._clean_table_formatting(self._convert_to_three_column_table(existing_table))
            else:
                return self._clean_table_formatting(self._convert_to_three_column_table(content.strip()))
        else:
            # Convert any text response to 3-column table format
            return self._clean_table_formatting(self._convert_any_text_to_three_column_table(content))
    
    def _clean_table_formatting(self, html_table: str) -> str:
        """Clean up table formatting and ensure proper spacing"""
        if not html_table:
            return html_table
            
        # Remove \n and excessive whitespace first
        cleaned = re.sub(r'\\n', '', html_table)
        cleaned = re.sub(r'\n\s*', '', cleaned)
        
        # Remove escaped quotes
        cleaned = cleaned.replace('\\"', '"')
        cleaned = cleaned.replace("\\'", "'")
        
        # Add proper spacing for readability
        cleaned = cleaned.replace('<table>', '<table> ')
        cleaned = cleaned.replace('<thead>', '<thead> ')
        cleaned = cleaned.replace('<tbody>', '<tbody> ')
        cleaned = cleaned.replace('<tr>', '<tr>')
        cleaned = cleaned.replace('</tr>', '</tr> ')
        cleaned = cleaned.replace('<td', ' <td')
        cleaned = cleaned.replace('<th', ' <th')
        cleaned = cleaned.replace('</thead>', ' </thead> ')
        cleaned = cleaned.replace('</tbody>', ' </tbody> ')
        cleaned = cleaned.replace('</table>', ' </table>')
        
        # Clean up multiple spaces
        cleaned = re.sub(r'\s+', ' ', cleaned)
        cleaned = cleaned.strip()
        
        # Ensure proper line structure (but still in one line)
        # This makes it more readable while keeping it as single line
        return cleaned
    
    def _extract_sources_comprehensive(self, content: str, citations: List, full_response: Dict) -> List[str]:
        """Extract sources using multiple methods"""
        sources = []
        
        # Method 1: Extract from citations field if available
        if citations:
            for citation in citations:
                if isinstance(citation, dict):
                    if "url" in citation:
                        title = citation.get("title", "")
                        url = citation["url"]
                        if title:
                            sources.append(f"{title}: {url}")
                        else:
                            sources.append(self._format_source_from_url(url))
                    elif "source" in citation:
                        sources.append(str(citation["source"]))
                elif isinstance(citation, str):
                    sources.append(citation)
        
        # Method 2: Look for citation patterns in content [1], [2], etc.
        citation_pattern = r'\[(\d+)\]'
        citation_numbers = re.findall(citation_pattern, content)
        
        if citation_numbers and "sources" in full_response:
            # Sometimes sources are in a separate field
            api_sources = full_response["sources"]
            for i, num in enumerate(citation_numbers):
                if i < len(api_sources):
                    sources.append(api_sources[i])
        
        # Method 3: Extract URLs directly from content
        url_sources = self._extract_urls_from_content(content)
        sources.extend(url_sources)
        
        # Method 4: Look for sources in the full API response
        if "web_results" in full_response:
            for result in full_response["web_results"]:
                if isinstance(result, dict) and "url" in result:
                    title = result.get("name", result.get("title", ""))
                    url = result["url"]
                    if title:
                        sources.append(f"{title}: {url}")
                    else:
                        sources.append(self._format_source_from_url(url))
        
        # Remove duplicates while preserving order
        seen = set()
        unique_sources = []
        for source in sources:
            if source not in seen:
                seen.add(source)
                unique_sources.append(source)
        
        # If no sources found, add a note
        if not unique_sources:
            unique_sources.append("Sources: Information compiled from real estate databases")
        
        return unique_sources
    
    def _extract_urls_from_content(self, content: str) -> List[str]:
        """Extract URLs from content and format them"""
        sources = []
        
        # Look for URLs in the content
        url_pattern = r'https?://[^\s<>"{}|\\^`\[\]\)]+(?:[^\s.,;!?<>"{}|\\^`\[\]\)])'
        urls = re.findall(url_pattern, content)
        
        for url in urls:
            # Clean up URL (remove any trailing punctuation)
            url = re.sub(r'[.,;!?]+$', '', url)
            sources.append(self._format_source_from_url(url))
        
        return sources
    
    def _format_source_from_url(self, url: str) -> str:
        """Format a URL into a readable source"""
        url = url.strip()
        
        if 'realtor.com' in url:
            return f"Realtor.com: {url}"
        elif 'zillow.com' in url:
            return f"Zillow: {url}"
        elif 'trulia.com' in url:
            return f"Trulia: {url}"
        elif 'redfin.com' in url:
            return f"Redfin: {url}"
        elif 'loopnet.com' in url:
            return f"LoopNet: {url}"
        elif 'apartments.com' in url:
            return f"Apartments.com: {url}"
        elif 'rentals.com' in url:
            return f"Rentals.com: {url}"
        elif 'propertyrecords.com' in url:
            return f"PropertyRecords.com: {url}"
        else:
            # Extract domain name for unknown sites
            domain_match = re.search(r'://(?:www\.)?([^/]+)', url)
            if domain_match:
                domain = domain_match.group(1).split('.')[0].title()
                return f"{domain}: {url}"
            else:
                return f"Source: {url}"
    
    def _convert_to_three_column_table(self, table_html: str) -> str:
        """Convert existing table to 3-column format"""
        # Extract data from existing table
        property_data = {}
        
        # Parse table rows
        row_pattern = r'<tr><td>([^<]+)</td><td>([^<]+)</td></tr>'
        matches = re.findall(row_pattern, table_html)
        
        for key, value in matches:
            property_data[key.strip()] = value.strip()
        
        return self._build_three_column_table(property_data)
    
    def _convert_any_text_to_three_column_table(self, text: str) -> str:
        """Convert any text response to 3-column table format"""
        if not isinstance(text, str):
            text = str(text)
        
        # Extract key property information from text using simple parsing
        property_data = {}
        
        # Extract address
        address_match = re.search(r'(\d+\s+[^,]+,\s*[^,]+,\s*[A-Z]{2}\s*\d{5})', text)
        if address_match:
            property_data['Address'] = address_match.group(1)
        
        # Extract price/value
        price_matches = re.findall(r'\$[\d,]+', text)
        if price_matches:
            property_data['Current Market Value'] = price_matches[0]
            if len(price_matches) > 1:
                property_data['Last Sale Price'] = price_matches[-1]
        
        # Extract year
        year_match = re.search(r'\b(19|20)\d{2}\b', text)
        if year_match:
            property_data['Year Built'] = year_match.group(0)
        
        # Extract square footage
        sqft_match = re.search(r'(\d+[\d,]*)\s*sq\.?\s*ft\.?', text, re.IGNORECASE)
        if sqft_match:
            property_data['Square Footage'] = f"{sqft_match.group(1)} sq ft"
        
        # Extract bedrooms/bathrooms
        bed_match = re.search(r'(\d+)[\s-]*(bed|br)', text, re.IGNORECASE)
        if bed_match:
            property_data['Bedrooms'] = bed_match.group(1)
        
        bath_match = re.search(r'(\d+(?:\.\d+)?)[\s-]*(bath|ba)', text, re.IGNORECASE)
        if bath_match:
            property_data['Bathrooms'] = bath_match.group(1)
        
        # Extract property type
        if 'single-family' in text.lower():
            property_data['Property Type'] = 'Single Family Residence'
        elif 'condo' in text.lower():
            property_data['Property Type'] = 'Condominium'
        elif 'gas station' in text.lower():
            property_data['Property Type'] = 'Commercial Gas Station'
        elif 'commercial' in text.lower():
            property_data['Property Type'] = 'Commercial'
        
        # Extract garage info
        garage_match = re.search(r'(\d+)[\s-]*car\s+garage', text, re.IGNORECASE)
        if garage_match:
            property_data['Parking'] = f"{garage_match.group(1)}-car garage"
        
        # If no specific info found, add the original text to general information
        if not property_data:
            property_data['General Information'] = text
        
        return self._build_three_column_table(property_data)
    
    def _build_three_column_table(self, property_data: dict) -> str:
        """Build 3-column HTML table from property data"""
        
        # Define which fields go in which category with their order
        details_fields = [
            'Address', 'Current Market Value', 'Last Sale Price', 'Property Type',
            'Square Footage', 'Lot Size', 'Year Built', 'Bedrooms', 'Bathrooms',
            'Zoning', 'Owner/Landlord', 'Property Tax'
        ]
        
        features_fields = [
            'Parking', 'Special Features', 'Building Condition', 'Heating/Cooling',
            'Flooring', 'Kitchen Features', 'Basement/Storage', 'Exterior Features',
            'Recent Updates'
        ]
        
        additional_fields = [
            'Neighborhood', 'Schools', 'Transportation', 'Local Amenities',
            'Market Trends', 'Investment Potential', 'Safety & Crime',
            'Environmental', 'HOA Information', 'Utilities', 'General Information'
        ]
        
        html = '<table>\n<thead>\n<tr><th>Category</th><th>Detail</th><th>Value</th></tr>\n</thead>\n<tbody>\n'
        
        # Property Details Section
        details_rows = []
        for field in details_fields:
            if field in property_data:
                escaped_value = str(property_data[field]).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                details_rows.append(f'<tr><td>{field}</td><td>{escaped_value}</td></tr>')
        
        if details_rows:
            # Add rowspan to first row
            rowspan = len(details_rows)
            html += f'<tr><td rowspan="{rowspan}">Property Details</td>' + details_rows[0][4:] + '\n'
            for row in details_rows[1:]:
                # Remove the first <td> from subsequent rows since we're using rowspan
                html += row.replace('<tr><td>', '<tr><td>').replace('</td><td>', '</td><td>', 1)[4:] + '\n'
        
        # Property Features Section
        features_rows = []
        for field in features_fields:
            if field in property_data:
                escaped_value = str(property_data[field]).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                features_rows.append(f'<tr><td>{field}</td><td>{escaped_value}</td></tr>')
        
        if features_rows:
            # Add rowspan to first row
            rowspan = len(features_rows)
            html += f'<tr><td rowspan="{rowspan}">Property Features</td>' + features_rows[0][4:] + '\n'
            for row in features_rows[1:]:
                html += row.replace('<tr><td>', '<tr><td>').replace('</td><td>', '</td><td>', 1)[4:] + '\n'
        else:
            html += '<tr><td rowspan="1">Property Features</td><td>Information not available</td><td>-</td></tr>\n'
        
        # Additional Information Section
        additional_rows = []
        for field in additional_fields:
            if field in property_data:
                escaped_value = str(property_data[field]).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                additional_rows.append(f'<tr><td>{field}</td><td>{escaped_value}</td></tr>')
        
        if additional_rows:
            # Add rowspan to first row
            rowspan = len(additional_rows)
            html += f'<tr><td rowspan="{rowspan}">Additional Information</td>' + additional_rows[0][4:] + '\n'
            for row in additional_rows[1:]:
                html += row.replace('<tr><td>', '<tr><td>').replace('</td><td>', '</td><td>', 1)[4:] + '\n'
        else:
            html += '<tr><td rowspan="1">Additional Information</td><td>Information not available</td><td>-</td></tr>\n'
        
        html += '</tbody>\n</table>'
        return html
    
    def _create_error_response(self, error_msg: str) -> Dict[str, Any]:
        """Create standardized error response in 3-column table format"""
        error_html = f"""<table>
<thead>
<tr><th>Category</th><th>Detail</th><th>Value</th></tr>
</thead>
<tbody>
<tr><td rowspan="3">Error Information</td><td>Error Message</td><td>{error_msg}</td></tr>
<tr><td>Status</td><td>Failed to retrieve property information</td></tr>
<tr><td>Suggestion</td><td>Please try again or check the address format</td></tr>
</tbody>
</table>"""
        
        return {
            "html_table": error_html,
            "sources": [],
            "suggested_questions": [
                "What property information is available for this address?",
                "How can I find accurate property details?",
                "What are the best real estate data sources?"
            ],
            "summary": f"Error: {error_msg}",
            "raw_response": error_msg,
            "conversation_context": 0
        }
    
    def save_html_file(self, html_table: str, filename: str = "property_report.html") -> None:
        """Save HTML table to file"""
        html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>Property Report</title>
    <meta charset="UTF-8">
</head>
<body>
    <h1>Property Information</h1>
    {html_table}
</body>
</html>"""
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(html_content)
        print(f"Saved to {filename}")

# Create client instance
perplexity_client = PerplexityClient()

def get_property_info(address: str, user_id: str = None) -> Dict[str, Any]:
    """
    SYNCHRONOUS function to get property information as HTML table
    
    Args:
        address (str): Property address to search
        user_id (str): Optional user ID to maintain conversation history
        
    Returns:
        Dict[str, Any]: Property information with HTML table, sources, suggested questions, and conversation context
    """
    question = f"Find detailed property information for: {address}"
    return perplexity_client.ask_question(question, user_id)

# Alternative method: Ask for sources explicitly
def get_property_info_with_explicit_sources(address: str, user_id: str = None) -> Dict[str, Any]:
    """
    Alternative method that explicitly asks for sources in the question
    """
    question = f"""Find detailed property information for: {address}
    
    Please include your sources and citations in the response."""
    
    result = perplexity_client.ask_question(question, user_id)
    
    # If still no sources, try to extract from the response text
    if not result["sources"] or len(result["sources"]) == 1:
        # Look for "Sources:" or "References:" in the raw response
        raw_response = result.get("raw_response", "")
        
        # Try to find sources section in the text
        sources_match = re.search(r'(?:Sources?|References?):\s*(.+?)(?:\n\n|\Z)', raw_response, re.DOTALL | re.IGNORECASE)
        if sources_match:
            sources_text = sources_match.group(1)
            # Split by lines and clean up
            additional_sources = [s.strip() for s in sources_text.split('\n') if s.strip()]
            result["sources"].extend(additional_sources)
    
    return result


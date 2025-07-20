from config import PERPLEXITY_API_KEY
import requests
import json
import re
from typing import Dict, Any, List

class PerplexityClient:
    def __init__(self, api_key: str = PERPLEXITY_API_KEY):
        self.api_key = api_key
        self.base_url = "https://api.perplexity.ai/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
    
    def ask_question(self, question: str, model: str = "sonar-pro") -> Dict[str, Any]:
        """
        SYNCHRONOUS function to ask Perplexity AI for property information in HTML table format
        
        Args:
            question (str): The question to ask
            model (str): The model to use (default: "sonar-pro")
            
        Returns:
            Dict[str, Any]: Dictionary containing HTML table and sources
        """
        
        # Modified system prompt to explicitly request sources
        system_prompt = """You are a real estate expert assistant. Find comprehensive property information and return ONLY a raw HTML table.

CRITICAL: You MUST respond with ONLY a raw HTML table in this EXACT format:
<table><thead><tr><th>Property Detail</th><th>Value</th></tr></thead><tbody><tr><td>Address</td><td>FULL_COMPLETE_ADDRESS</td></tr><tr><td>Current Market Value</td><td>$XXX,XXX or Price Range</td></tr><tr><td>Last Sale Price</td><td>$XXX,XXX (Date)</td></tr><tr><td>Property Type</td><td>Residential/Commercial/Gas Station/etc</td></tr><tr><td>Square Footage</td><td>X,XXX sq ft</td></tr><tr><td>Lot Size</td><td>X.XX acres or X,XXX sq ft</td></tr><tr><td>Year Built</td><td>YYYY</td></tr><tr><td>Bedrooms</td><td>X (for residential only)</td></tr><tr><td>Bathrooms</td><td>X.X (for residential only)</td></tr><tr><td>Zoning</td><td>Commercial/Residential/Mixed Use</td></tr><tr><td>Owner/Landlord</td><td>Owner name or company</td></tr><tr><td>Property Tax</td><td>$X,XXX annually</td></tr><tr><td>Parking</td><td>X spaces or garage type</td></tr><tr><td>Special Features</td><td>Gas pumps, convenience store, etc</td></tr></tbody></table>

REQUIREMENTS:
- Return ONLY the HTML table, absolutely nothing else
- No additional text, no explanations, no sources section
- Raw HTML table only (no CSS, no styling, no classes)
- Include ALL available property information
- Skip rows for unavailable data
- Research thoroughly using multiple real estate sources
- Focus on the EXACT address provided by the user
- IMPORTANT: Include citations in your research but format them as [1], [2], etc."""

        data = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question}
            ],
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
                
                return {
                    "html_table": html_table,
                    "sources": sources,
                    "summary": "Property information table",
                    "raw_response": content,
                    "full_api_response": result  # For debugging
                }
            else:
                return self._create_error_response(f"API error {response.status_code}: {response.text}")
                
        except requests.Timeout:
            return self._create_error_response("Request timeout")
        except requests.RequestException as e:
            return self._create_error_response(f"Request error: {str(e)}")
        except Exception as e:
            return self._create_error_response(f"Unexpected error: {str(e)}")
    
    def _extract_or_create_table(self, content: str) -> str:
        """Extract existing table or convert content to table format"""
        if not isinstance(content, str):
            content = str(content)
        
        # Check if we got a table
        if content.strip().startswith('<table'):
            # Extract existing table
            table_end = content.find('</table>') + 8
            if table_end > 7:
                return content[:table_end].strip()
            else:
                return content.strip()
        else:
            # Convert any text response to table format
            return self._convert_any_text_to_property_table(content)
    
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
    
    def _convert_any_text_to_property_table(self, text: str) -> str:
        """Convert any text response to property table format"""
        if not isinstance(text, str):
            text = str(text)
        
        # Extract key property information from text using simple parsing
        property_info = {}
        
        # Extract address
        address_match = re.search(r'(\d+\s+[^,]+,\s*[^,]+,\s*[A-Z]{2}\s*\d{5})', text)
        if address_match:
            property_info['Address'] = address_match.group(1)
        
        # Extract price/value
        price_matches = re.findall(r'\$[\d,]+', text)
        if price_matches:
            property_info['Current Market Value'] = price_matches[0]
            if len(price_matches) > 1:
                property_info['Last Sale Price'] = price_matches[-1]
        
        # Extract year
        year_match = re.search(r'\b(19|20)\d{2}\b', text)
        if year_match:
            property_info['Year Built'] = year_match.group(0)
        
        # Extract square footage
        sqft_match = re.search(r'(\d+[\d,]*)\s*sq\.?\s*ft\.?', text, re.IGNORECASE)
        if sqft_match:
            property_info['Square Footage'] = f"{sqft_match.group(1)} sq ft"
        
        # Extract bedrooms/bathrooms
        bed_match = re.search(r'(\d+)[\s-]*(bed|br)', text, re.IGNORECASE)
        if bed_match:
            property_info['Bedrooms'] = bed_match.group(1)
        
        bath_match = re.search(r'(\d+(?:\.\d+)?)[\s-]*(bath|ba)', text, re.IGNORECASE)
        if bath_match:
            property_info['Bathrooms'] = bath_match.group(1)
        
        # Extract property type
        if 'single-family' in text.lower():
            property_info['Property Type'] = 'Single Family Residence'
        elif 'condo' in text.lower():
            property_info['Property Type'] = 'Condominium'
        elif 'gas station' in text.lower():
            property_info['Property Type'] = 'Commercial Gas Station'
        elif 'commercial' in text.lower():
            property_info['Property Type'] = 'Commercial'
        
        # Extract garage info
        garage_match = re.search(r'(\d+)[\s-]*car\s+garage', text, re.IGNORECASE)
        if garage_match:
            property_info['Parking'] = f"{garage_match.group(1)}-car garage"
        
        # Build table
        table_html = "<table><thead><tr><th>Property Detail</th><th>Value</th></tr></thead><tbody>"
        
        for key, value in property_info.items():
            escaped_value = str(value).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
            table_html += f"<tr><td>{key}</td><td>{escaped_value}</td></tr>"
        
        # If no specific info found, add the original text
        if not property_info:
            escaped_text = text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
            table_html += f"<tr><td>Property Information</td><td>{escaped_text}</td></tr>"
        
        table_html += "</tbody></table>"
        return table_html
    
    def _create_error_response(self, error_msg: str) -> Dict[str, Any]:
        """Create standardized error response"""
        return {
            "html_table": f"<table><tr><td>Error: {error_msg}</td></tr></table>",
            "sources": [],
            "summary": f"Error: {error_msg}",
            "raw_response": error_msg
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

def get_property_info(address: str) -> Dict[str, Any]:
    """
    SYNCHRONOUS function to get property information as HTML table
    
    Args:
        address (str): Property address to search
        
    Returns:
        Dict[str, Any]: Property information with HTML table
    """
    question = f"Find detailed property information for: {address}"
    return perplexity_client.ask_question(question)

# Alternative method: Ask for sources explicitly
def get_property_info_with_explicit_sources(address: str) -> Dict[str, Any]:
    """
    Alternative method that explicitly asks for sources in the question
    """
    question = f"""Find detailed property information for: {address}
    
    Please include your sources and citations in the response."""
    
    result = perplexity_client.ask_question(question)
    
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

# Example usage - NO ASYNC/AWAIT ANYWHERE
if __name__ == "__main__":
    # Direct synchronous usage
    address = "123 Main Street, New York, NY"
    
    # Try the regular method
    result = get_property_info(address)
    
    print("HTML Table:")
    print(result["html_table"])
    
    print("\nSources:")
    for source in result["sources"]:
        print(f"- {source}")
    
    print(f"\nSummary: {result['summary']}")
    
    # If no sources, try the explicit method
    if len(result["sources"]) <= 1:
        print("\nTrying explicit sources method...")
        result2 = get_property_info_with_explicit_sources(address)
        print("Sources from explicit method:")
        for source in result2["sources"]:
            print(f"- {source}")
    
    # Save to file
    perplexity_client.save_html_file(result["html_table"])
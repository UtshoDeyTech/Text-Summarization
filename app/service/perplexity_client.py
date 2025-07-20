from config import PERPLEXITY_API_KEY
import requests
import json
from typing import Dict, Any

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
        
        system_prompt = """You are a real estate expert assistant. Find comprehensive property information and return a raw HTML table followed by source URLs.

CRITICAL: You MUST respond in this EXACT format:
<table><thead><tr><th>Property Detail</th><th>Value</th></tr></thead><tbody><tr><td>Address</td><td>FULL_COMPLETE_ADDRESS</td></tr><tr><td>Current Market Value</td><td>$XXX,XXX or Price Range</td></tr><tr><td>Last Sale Price</td><td>$XXX,XXX (Date)</td></tr><tr><td>Property Type</td><td>Residential/Commercial/Gas Station/etc</td></tr><tr><td>Square Footage</td><td>X,XXX sq ft</td></tr><tr><td>Lot Size</td><td>X.XX acres or X,XXX sq ft</td></tr><tr><td>Year Built</td><td>YYYY</td></tr><tr><td>Bedrooms</td><td>X (for residential only)</td></tr><tr><td>Bathrooms</td><td>X.X (for residential only)</td></tr><tr><td>Zoning</td><td>Commercial/Residential/Mixed Use</td></tr><tr><td>Owner/Landlord</td><td>Owner name or company</td></tr><tr><td>Property Tax</td><td>$X,XXX annually</td></tr><tr><td>Rental Income</td><td>$X,XXX/month (if rental)</td></tr><tr><td>Parking</td><td>X spaces</td></tr><tr><td>Special Features</td><td>Gas pumps, convenience store, etc</td></tr></tbody></table>

SOURCES:
https://www.example1.com
https://www.example2.com
https://www.example3.com

PROPERTY INFORMATION TO RESEARCH AND INCLUDE:
- Complete address with ZIP code
- Current market value or estimated value
- Recent sale price and date
- Property type (residential, commercial, gas station, retail, etc.)
- Building square footage
- Lot size in acres or square feet
- Year of construction
- For residential: bedrooms, bathrooms
- For commercial: business type, tenant information
- Zoning classification
- Property owner information
- Annual property taxes
- Rental income (if applicable)
- Parking availability
- Special features (for gas stations: number of pumps, convenience store, car wash, etc.)

REQUIREMENTS:
- First provide the HTML table
- Then add "SOURCES:" on a new line
- Then list actual URLs you used for research (one per line)
- Raw HTML table only (no CSS, no styling, no classes)
- Include ALL available information from the list above
- Skip rows for unavailable data (don't include "Not Available" rows)
- For gas stations specifically, include: pump count, fuel types, convenience store details, car wash facilities
- For commercial properties, include: tenant information, lease details, business type
- Research thoroughly using multiple real estate sources like Zillow, LoopNet, Realtor.com, etc."""

        data = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question}
            ]
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
                
                # Extract content
                content = None
                if "choices" in result and len(result["choices"]) > 0:
                    choice = result["choices"][0]
                    if "message" in choice and "content" in choice["message"]:
                        content = choice["message"]["content"]
                
                if not content:
                    return self._create_error_response("No content in API response")
                
                # Parse response to extract HTML table and sources
                if content.strip().startswith('<table') and 'SOURCES:' in content:
                    # Split content into table and sources
                    parts = content.split('SOURCES:')
                    html_table = parts[0].strip()
                    
                    # Extract source URLs
                    sources = []
                    if len(parts) > 1:
                        source_lines = parts[1].strip().split('\n')
                        for line in source_lines:
                            line = line.strip()
                            if line and (line.startswith('http') or line.startswith('www')):
                                sources.append(line)
                    
                    return {
                        "html_table": html_table,
                        "sources": sources,
                        "summary": "Property information table with sources",
                        "raw_response": content
                    }
                elif content.strip().startswith('<table') and content.strip().endswith('</table>'):
                    # Direct HTML table response without sources
                    return {
                        "html_table": content.strip(),
                        "sources": [],
                        "summary": "Property information table",
                        "raw_response": content
                    }
                else:
                    # Try to parse as JSON (fallback)
                    try:
                        parsed = json.loads(content)
                        return {
                            "html_table": parsed.get("html_table", ""),
                            "sources": parsed.get("source", []),
                            "summary": parsed.get("summary", ""),
                            "raw_response": content
                        }
                    except json.JSONDecodeError:
                        # Convert text to table as final fallback
                        return {
                            "html_table": self._text_to_table(content),
                            "sources": [],
                            "summary": content,
                            "raw_response": content
                        }
            else:
                return self._create_error_response(f"API error {response.status_code}: {response.text}")
                
        except requests.Timeout:
            return self._create_error_response("Request timeout")
        except requests.RequestException as e:
            return self._create_error_response(f"Request error: {str(e)}")
        except Exception as e:
            return self._create_error_response(f"Unexpected error: {str(e)}")
    
    def _text_to_table(self, text: str) -> str:
        """Convert text to basic HTML table"""
        if not isinstance(text, str):
            text = str(text)
        
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        
        table = "<table><thead><tr><th>Information</th></tr></thead><tbody>"
        for line in lines:
            escaped = line.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
            table += f"<tr><td>{escaped}</td></tr>"
        table += "</tbody></table>"
        
        return table
    
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
    question = f"Find detailed property information for: {address}"
    return perplexity_client.ask_question(question)


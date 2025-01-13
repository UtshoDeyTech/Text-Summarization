from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import openai
import re
from typing import Dict
from bs4 import BeautifulSoup
from config import OPENAI_API_KEY

router = APIRouter()
openai.api_key = OPENAI_API_KEY

class EmailContent(BaseModel):
    subject: str
    body: str
    tail: str

class EmailRequest(BaseModel):
    content: str

def clean_html_to_text(html_content: str) -> Dict[str, str]:
    """Convert HTML table to structured text format"""
    # Parse HTML
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Initialize sections
    sections = {
        'Policy Information': [],
        'Coverage Details': [],
        'Additional Information': []
    }
    
    current_section = None
    
    # Process each row
    for row in soup.find_all('tr'):
        # Check if it's a section header
        header = row.find('th', {'colspan': '2'})
        if header and header.text in sections:
            current_section = header.text
            continue
            
        # Process regular rows
        if current_section and len(row.find_all(['td', 'th'])) == 2:
            cells = row.find_all(['td', 'th'])
            if cells[0].text != "Field":  # Skip the initial header row
                sections[current_section].append(f"{cells[0].text}: {cells[1].text}")
    
    # Convert to formatted text
    formatted_text = ""
    for section, items in sections.items():
        if items:
            formatted_text += f"\n{section}:\n"
            formatted_text += "\n".join(f"- {item}" for item in items)
            formatted_text += "\n"
    
    # Extract customer name if present
    customer_name = ""
    for item in sections['Policy Information']:
        if "Insured:" in item:
            customer_name = item.split(": ")[1]
            break
    
    return {
        "formatted_text": formatted_text.strip(),
        "customer_name": customer_name
    }

def generate_email_content(formatted_data: Dict[str, str]) -> Dict[str, str]:
    """Generate email content using OpenAI"""
    system_prompt = """You are an insurance company representative. Generate a concise and professional email summary based on the policy information provided. The email should be brief but informative.

You must return a valid JSON object with exactly these three fields:
{
    "subject": "Your email subject line",
    "body": "Your email body content (without any signature)",
    "tail": "Sincerely,\\n[Your Name]\\n[Your Title]\\n[Insurance Company Name]"
}

Requirements:
1. Response must be a valid JSON object
2. All three fields (subject, body, tail) must be present
3. The tail field must be exactly as shown above
4. Do not include the signature in the body
5. No additional fields or formatting
"""

    user_prompt = f"""Generate a policy summary email using this information:

{formatted_data['formatted_text']}

Requirements:
1. Use customer name: {formatted_data['customer_name']}
2. Focus on these key elements:
   - Primary dwelling coverage
   - Liability coverage
3. Format requirements:
   - Use bullet points (•) for lists
   - Keep paragraphs short and focused
   - Maintain professional tone
   - Include policy number in subject
4. Avoid:
   - Technical insurance jargon
   - Excessive details
   - Marketing language
   
   
**Donot add Total premium in your email.**
   """

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error generating email: {str(e)}"
        )

@router.post("/generate-email")
async def generate_email(request: EmailRequest) -> EmailContent:
    try:
        # Clean and format the HTML content
        formatted_data = clean_html_to_text(request.content)
        
        # Generate email content
        email_json = generate_email_content(formatted_data)
        
        # Parse the JSON string response from OpenAI
        import json
        try:
            # Clean up any potential formatting issues
            email_json = email_json.strip()
            if not email_json.startswith('{'):
                raise ValueError("Response is not in JSON format")
                
            email_content = json.loads(email_json)
            
            # Ensure all required fields are present
            required_fields = {'subject', 'body', 'tail'}
            if not all(field in email_content for field in required_fields):
                raise ValueError(f"Missing required fields. Got: {list(email_content.keys())}")
            
            return EmailContent(
                subject=email_content['subject'],
                body=email_content['body'],
                tail=email_content['tail']
            )
        except json.JSONDecodeError as e:
            raise HTTPException(
                status_code=500,
                detail=f"Invalid JSON response from OpenAI: {str(e)}\nResponse: {email_json}"
            )
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Error processing OpenAI response: {str(e)}\nResponse: {email_json}"
            )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing request: {str(e)}"
        )
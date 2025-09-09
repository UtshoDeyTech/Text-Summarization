from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from typing import Optional, Dict, Any
from datetime import datetime
from PyPDF2 import PdfReader
from docx import Document
from io import BytesIO
from openai import AsyncOpenAI
import json
import re
from app.service.log_client import logger
from config import OPENAI_API_KEY, MODEL

router = APIRouter()
client = AsyncOpenAI(api_key=OPENAI_API_KEY)

SUPPORTED_EXTENSIONS = {
    'pdf': 'application/pdf',
    'docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
}

def extract_text_from_pdf(file_obj):
    """Extract text from PDF file"""
    try:
        pdf = PdfReader(file_obj)
        text = ""
        for page in pdf.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        logger.error(f"PDF text extraction failed | error={str(e)}")
        raise

def extract_text_from_docx(file_obj):
    """Extract text from DOCX file"""
    try:
        doc = Document(file_obj)
        text = ""
        for para in doc.paragraphs:
            text += para.text + "\n"
        return text
    except Exception as e:
        logger.error(f"DOCX text extraction failed | error={str(e)}")
        raise

def create_extraction_prompt(text: str) -> str:
    """Create a prompt for OpenAI to extract specific fields from insurance document text"""
    return f"""
You are an expert at extracting information from insurance documents. Please analyze the following document text and extract the specified information. Return the result as a valid JSON object with the exact field names specified.

Extract the following fields:
1. Date - Any date mentioned in the document (format as MM/DD/YYYY if possible)
2. AGENCY_Name - The name of the insurance agency
3. Agency_Phone - Primary agency phone number (format: numbers only or with standard formatting)
4. Agency_Email - Agency email address
5. Primary_Phone - Primary contact phone number
6. Secondary_Phone - Secondary contact phone number (if available)
7. Form_Name - The name/type of the insurance form
8. Contact_Name - Full name of the primary contact person (First, Middle, Last if available)

Document Text:
{text}

Instructions:
- If a field is not found, set its value to null
- For phone numbers, extract the full number including area code if available
- For names, combine first, middle, and last names into a single field
- For dates, use MM/DD/YYYY format if possible
- Return only valid JSON format
- Be as accurate as possible in extraction

Expected JSON format:
{{
    "Date": "MM/DD/YYYY or null",
    "AGENCY_Name": "agency name or null",
    "Agency_Phone": "phone number or null",
    "Agency_Email": "email or null", 
    "Primary_Phone": "phone number or null",
    "Secondary_Phone": "phone number or null",
    "Form_Name": "form name or null",
    "Contact_Name": "full name or null"
}}
"""

async def extract_fields_with_openai(text: str) -> Dict[str, Any]:
    """Use OpenAI to extract specific fields from the document text"""
    try:
        prompt = create_extraction_prompt(text)
        
        response = await client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "system", 
                    "content": "You are an expert document parser specializing in insurance forms. Extract information accurately and return valid JSON only."
                },
                {
                    "role": "user", 
                    "content": prompt
                }
            ],
            temperature=0.1,  # Low temperature for consistent extraction
            max_tokens=500
        )
        
        # Extract the JSON from the response
        content = response.choices[0].message.content.strip()
        
        # Try to parse JSON directly
        try:
            extracted_data = json.loads(content)
            return extracted_data
        except json.JSONDecodeError:
            # If direct parsing fails, try to extract JSON from the response
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                extracted_data = json.loads(json_match.group())
                return extracted_data
            else:
                raise ValueError("No valid JSON found in OpenAI response")
                
    except Exception as e:
        logger.error(f"OpenAI extraction failed | error={str(e)}")
        raise

def validate_extracted_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and clean the extracted data, excluding null values"""
    # Basic fields
    basic_fields = [
        "Date", "AGENCY_Name", "Agency_Phone", "Agency_Email",
        "Primary_Phone", "Secondary_Phone", "Form_Name", "Contact_Name"
    ]
    
    # Package Policy Information fields
    package_policy_fields = [
        "Proposed_Effective_Date", "Proposed_Expiration_Date", "Billing_Plan",
        "Payment_Plan", "Audit"
    ]
    
    # Applicant Information fields
    applicant_info_fields = [
        "First_Named_Insured", "Mailing_Address", "FEIN_SSN", "Applicant_Phone",
        "Applicant_Email", "Website_Address", "Number_of_Employees", "Annual_Revenues",
        "Business_Entity_Type", "Date_Business_Started"
    ]
    
    # Premises Information fields
    premises_info_fields = [
        "Location_Number", "Building_Number", "Street_Address", "City", "County",
        "State", "ZIP_Code", "City_Limits", "Interest_Type", "Year_Built",
        "Percent_Occupied", "Nature_of_Business"
    ]
    
    def is_valid_value(value):
        """Check if value is valid (not null, not empty string, not 'null' string)"""
        if value is None:
            return False
        if isinstance(value, str):
            cleaned = value.strip().lower()
            return cleaned != "" and cleaned != "null" and cleaned != "not found"
        return True
    
    def clean_data(obj):
        """Clean data and remove null/empty values"""
        if isinstance(obj, dict):
            cleaned = {}
            for key, value in obj.items():
                cleaned_value = clean_data(value)
                if is_valid_value(cleaned_value):
                    cleaned[key] = cleaned_value
            return cleaned if cleaned else None
        elif isinstance(obj, str):
            cleaned = obj.strip()
            return cleaned if cleaned and cleaned.lower() not in ["null", "not found", ""] else None
        else:
            return obj if obj is not None else None
    
    validated_data = {}
    
    # Validate basic fields
    for field in basic_fields:
        value = clean_data(data.get(field))
        if is_valid_value(value):
            validated_data[field] = value
    
    # Validate Package Policy Information
    package_policy_data = data.get("Package_Policy_Information", {})
    package_section = {}
    for field in package_policy_fields:
        value = clean_data(package_policy_data.get(field))
        if is_valid_value(value):
            package_section[field] = value
    
    if package_section:  # Only include section if it has data
        validated_data["Package_Policy_Information"] = package_section
    
    # Validate Applicant Information
    applicant_data = data.get("Applicant_Information", {})
    applicant_section = {}
    for field in applicant_info_fields:
        value = clean_data(applicant_data.get(field))
        if is_valid_value(value):
            applicant_section[field] = value
    
    if applicant_section:  # Only include section if it has data
        validated_data["Applicant_Information"] = applicant_section
    
    # Validate Premises Information
    premises_data = data.get("Premises_Information", {})
    premises_section = {}
    for field in premises_info_fields:
        value = clean_data(premises_data.get(field))
        if is_valid_value(value):
            premises_section[field] = value
    
    if premises_section:  # Only include section if it has data
        validated_data["Premises_Information"] = premises_section
    
    return validated_data

@router.post("/convert_file_to_json")
async def convert_file_to_json(file: UploadFile = File(...)):
    """
    Convert insurance document (PDF/DOCX) to JSON with extracted fields
    """
    start_time = datetime.utcnow()
    
    try:
        # Validate file type
        if not file.filename:
            raise HTTPException(
                status_code=400,
                detail={
                    "status_code": "400",
                    "error_messages": ["No filename provided"]
                }
            )
        
        file_extension = file.filename.split('.')[-1].lower()
        
        if file_extension not in SUPPORTED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail={
                    "status_code": "400",
                    "error_messages": [f"Supported file types are: {', '.join(SUPPORTED_EXTENSIONS.keys())}"]
                }
            )
        
        # Read file content
        file_content = await file.read()
        file_obj = BytesIO(file_content)
        
        logger.info(f"Processing file for JSON conversion | filename={file.filename}, size={len(file_content)/1024/1024:.2f} MB")
        
        # Extract text based on file type
        if file_extension == 'pdf':
            extracted_text = extract_text_from_pdf(file_obj)
        elif file_extension == 'docx':
            extracted_text = extract_text_from_docx(file_obj)
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")
        
        if not extracted_text or not extracted_text.strip():
            raise HTTPException(
                status_code=400,
                detail={
                    "status_code": "400",
                    "error_messages": ["No text could be extracted from the document"]
                }
            )
        
        logger.info(f"Text extraction completed | text_length={len(extracted_text)} characters")
        
        # Extract fields using OpenAI
        logger.info("Starting field extraction with OpenAI")
        extracted_fields = await extract_fields_with_openai(extracted_text)
        
        # Validate and clean the extracted data
        validated_data = validate_extracted_data(extracted_fields)
        
        # Calculate processing time
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        
        logger.info(f"Document to JSON conversion completed | filename={file.filename}, processing_time={processing_time:.2f}s")
        
        # Return the response
        return JSONResponse(
            content={
                "status": "success",
                "filename": file.filename,
                "file_type": file_extension,
                "processing_time_seconds": round(processing_time, 2),
                "extracted_data": validated_data,
                "status_code": "200"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        error_msg = f"Document conversion failed | filename={file.filename if file and file.filename else 'unknown'}, error={str(e)}"
        logger.error(error_msg)
        
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        
        raise HTTPException(
            status_code=500,
            detail={
                "status_code": "500",
                "error_messages": [str(e)],
                "processing_time_seconds": round(processing_time, 2)
            }
        )

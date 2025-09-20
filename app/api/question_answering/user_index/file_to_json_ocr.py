from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from datetime import datetime
import fitz  # PyMuPDF
from app.service.log_client import logger

router = APIRouter()


@router.post("/extract_acord125")
async def extract_acord125(file: UploadFile = File(...)):
    """
    Extract Applicant and Premises information from an ACORD 125 PDF.
    If the format doesn't match (not an ACORD 125 or missing fields),
    return a valid structured response with status message.
    """

    try:
        # Validate file type
        if not file.filename.lower().endswith(".pdf"):
            return JSONResponse(content={
                "status": "error",
                "message": "Only PDF files are allowed"
            }, status_code=400)

        # Read file content
        pdf_content = await file.read()

        # Open PDF
        pdf_document = fitz.open(stream=pdf_content, filetype="pdf")
        if len(pdf_document) == 0:
            return JSONResponse(content={
                "status": "error",
                "message": "PDF appears to be empty"
            }, status_code=400)

        # Collect all form field values
        extracted_data = {}
        for page_num, page in enumerate(pdf_document, start=1):
            fields = page.widgets()
            if not fields:
                continue
            for widget in fields:
                if widget.field_name:
                    value = widget.field_value if widget.field_value else None
                    extracted_data[widget.field_name] = value

        pdf_document.close()

        if not extracted_data:
            return JSONResponse(content={
                "status": "error",
                "message": "No form fields found in PDF. Possibly not an ACORD 125 form."
            }, status_code=200)

        # Map fields of interest (using known ACORD 125 field names)
        applicant_info = {
            "name": extracted_data.get("NAME_First_Named_Insured__Other_Named_Insureds"),
            "mailing_address": extracted_data.get("MAILING_ADDRESS_INCL_ZIP4_of_First_Named_Insured"),
            "email": extracted_data.get("ADDRESS_E_MAIL"),
            "phone": extracted_data.get("AC_No_Ext_PHONE0")
        }

        premises_info = {
            "address": extracted_data.get("Textfield35"),
            "year_built": extracted_data.get("OWNER_TENANT")
        }

        # Check if required fields exist
        if not applicant_info["name"] or not premises_info["address"]:
            return JSONResponse(content={
                "status": "error",
                "message": "PDF does not match expected ACORD 125 format. Required fields missing.",
                "extracted_preview": {k: v for k, v in list(extracted_data.items())[:10]}  # first 10 fields for debug
            }, status_code=200)

        # Build final response
        result = {
            "applicant_information": applicant_info,
            "premises_information": premises_info,
            # "all_fields": extracted_data,
            "extraction_timestamp": datetime.now().isoformat()
        }

        logger.info(f"Extracted Applicant: {applicant_info}, Premises: {premises_info}")

        return JSONResponse(content={
            "status": "success",
            "message": "ACORD 125 fields extracted successfully",
            "data": result
        })

    except Exception as e:
        logger.error(f"Error processing ACORD 125 PDF: {str(e)}")
        return JSONResponse(content={
            "status": "error",
            "message": f"Error processing PDF: {str(e)}"
        }, status_code=500)

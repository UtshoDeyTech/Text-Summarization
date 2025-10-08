from fastapi import APIRouter, UploadFile, File
from fastapi.responses import JSONResponse
from datetime import datetime
import fitz  # PyMuPDF
from app.service.log_client import logger

router = APIRouter()

# Section keyword mapping for each ACORD form
ACORD_SECTIONS = {
    "125": [
        "APPLICANT INFORMATION",
        "PACKAGE POLICY INFORMATION",
        "PREMISES INFORMATION",
        "NATURE OF BUSINESS/DESCRIPTION OF OPERATIONS BY PREMISE"
    ],
    "126": [
        "COMMERCIAL GENERAL LIABILITY SECTION",
        "COVERAGES"
    ],
    "140": [
        "PROPERTY SECTION",
        "PREMISES INFORMATION"
    ]
}

# Field name mapping for each ACORD form
FIELD_NAME_MAPPING = {
    "125": {
        "Textfield20": "PROPOSED_EFF_DATE",
        "Textfield21": "PROPOSED_EXP_DATE",
        "Textfield33": "LOCATION_NUMBER",
        "Textfield34": "BUILDING_NUMBER",
        "Textfield35": "PREMISES_ADDRESS",
        "Textfield49": "BUSINESS_DESCRIPTION",
        "OWNER_TENANT": "YEAR_BUILT",
        "OWNER_TENANT2": "PERCENT_OCCUPIED",
        "AC_No_Ext_PHONE0": "INSPECTION_PHONE",
        "ADDRESS_E_MAIL": "INSPECTION_EMAIL"
    },
    "126": {
        "DATE0": "APPLICATION_DATE",
        "AC_No_Ext_PHONE2": "PHONE_NUMBER",
        "APPLICANT_Insured_Named_First": "INSURED_NAME",
        "Textfield154": "GENERAL_AGGREGATE_LIMIT",
        "Textfield156": "PRODUCTS_COMPLETED_OPS_AGGREGATE",
        "Textfield157": "PERSONAL_ADVERTISING_INJURY_LIMIT",
        "Textfield158": "EACH_OCCURRENCE_LIMIT",
        "Textfield159": "DAMAGE_TO_RENTED_PREMISES_LIMIT",
        "Textfield160": "MEDICAL_EXPENSE_LIMIT",
        "Textfield170": "LOCATION_NUMBER",
        "Textfield171": "CLASSIFICATION_TYPE",
        "Textfield172": "AREA_SQFT",
        "Textfield181": "REVENUE_TYPE",
        "Textfield182": "ANNUAL_REVENUE",
        "REMARKS1": "REMARKS"
    },
    "140": {
        "DATE_MMDDYYYY0": "APPLICATION_DATE",
        "AC_No_Ext_PHONE3": "PHONE_NUMBER",
        "AC_No_FAX": "FAX_NUMBER",
        "APPLICANT_Insured_Named_First0": "INSURED_NAME",
        "EFFECTIVE_DATE0": "EFFECTIVE_DATE",
        "EXPIRATION_DATE0": "EXPIRATION_DATE",
        "BUILDING0": "BUILDING_NUMBER",
        "BLDG_DESCRIPTION": "BUILDING_DESCRIPTION",
        "Textfield279": "PROPERTY_TYPE",
        "Textfield280": "BUILDING_VALUE",
        "Textfield281": "COINSURANCE_PERCENT",
        "Textfield282": "VALUATION_TYPE",
        "Textfield283": "CAUSE_OF_LOSS",
        "Textfield286": "DEDUCTIBLE",
        "Textfield288": "COVERAGE_FORM",
        "Textfield289": "EARTHQUAKE_COVERAGE",
        "Textfield290": "EARTHQUAKE_STATUS",
        "Textfield299": "BUSINESS_INCOME_TYPE",
        "Textfield300": "BUSINESS_INCOME_LIMIT",
        "Textfield302": "BUSINESS_INCOME_PERIOD",
        "CONSTRUCTION_TYPE": "CONSTRUCTION_TYPE",
        "STORIES": "NUMBER_OF_STORIES",
        "YR_BUILT": "YEAR_BUILT",
        "TOTAL_AREA": "TOTAL_AREA_SQFT",
        "BURGLAR_ALARM_TYPE": "BURGLAR_ALARM_TYPE",
        "PREMISES_FIRE_PROTECTION_Sprinklers_Standpipes_COC": "FIRE_PROTECTION_SYSTEM",
        "SPRNK": "SPRINKLER_COVERAGE_PERCENT"
    }
}


def match_section(form_number: str, text: str):
    """Return True if page text contains one of the desired sections."""
    sections = ACORD_SECTIONS.get(form_number, [])
    for section in sections:
        if section.lower() in text.lower():
            return True
    return False


def replace_field_names(form_number: str, data: dict) -> dict:
    """Replace technical field names with human-readable names based on mapping."""
    mapping = FIELD_NAME_MAPPING.get(form_number, {})
    if not mapping:
        return data
    
    replaced_data = {}
    for key, value in data.items():
        # Use mapped name if exists, otherwise keep original
        new_key = mapping.get(key, key)
        replaced_data[new_key] = value
    
    return replaced_data


@router.post("/extract_acord_forms")
async def extract_acord_forms(file: UploadFile = File(...)):
    """
    Extract only selected subsections from ACORD PDFs (125, 126, 140).
    Returns structure like {125: {...}, 126: {...}, 140: {...}}.
    """

    try:
        # Validate file type
        if not file.filename.lower().endswith(".pdf"):
            return JSONResponse(content={
                "status": "error",
                "message": "Only PDF files are allowed"
            }, status_code=400)

        pdf_content = await file.read()
        pdf_document = fitz.open(stream=pdf_content, filetype="pdf")

        if len(pdf_document) == 0:
            return JSONResponse(content={
                "status": "error",
                "message": "PDF appears to be empty"
            }, status_code=400)

        acord_data = {"125": {}, "126": {}, "140": {}}
        current_form = None

        for page_num, page in enumerate(pdf_document, start=1):
            text = page.get_text("text")

            # Detect ACORD form number
            if "ACORD 125" in text:
                current_form = "125"
            elif "ACORD 126" in text:
                current_form = "126"
            elif "ACORD 140" in text:
                current_form = "140"

            if not current_form:
                continue

            # Check if page belongs to one of the required subsections
            if not match_section(current_form, text):
                continue  # skip unrelated sections

            # Extract widget data (form fields)
            fields = page.widgets()
            if not fields:
                continue

            page_data = {}
            for widget in fields:
                if widget.field_name:
                    value = widget.field_value if widget.field_value else None
                    # Ignore empty, "Off", and "x" values
                    if value not in (None, "", "Off", "x"):
                        page_data[widget.field_name] = value

            if current_form and page_data:
                acord_data[current_form].update(page_data)

        pdf_document.close()

        # Replace field names with human-readable names
        for form_number in acord_data:
            if acord_data[form_number]:
                acord_data[form_number] = replace_field_names(form_number, acord_data[form_number])

        # Keep only non-empty form data
        acord_data = {k: v for k, v in acord_data.items() if v}

        if not acord_data:
            return JSONResponse(content={
                "status": "error",
                "message": "No matching ACORD sections found in the PDF."
            }, status_code=200)

        response_data = {
            "status": "success",
            "message": "Selected ACORD form sections extracted successfully",
            "data": acord_data,
            "extraction_timestamp": datetime.now().isoformat()
        }

        logger.info(f"Extracted ACORD sections: {list(acord_data.keys())}")
        return JSONResponse(content=response_data, status_code=200)

    except Exception as e:
        logger.error(f"Error processing ACORD PDF: {str(e)}")
        return JSONResponse(content={
            "status": "error",
            "message": f"Error processing PDF: {str(e)}"
        }, status_code=500)
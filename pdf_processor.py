from PyPDF2 import PdfReader

def process_pdf(uploaded_pdf):
    """Extract text from the uploaded PDF file."""
    pdfreader = PdfReader(uploaded_pdf)
    raw_text = ""
    for page in pdfreader.pages:
        content = page.extract_text()
        if content:
            raw_text += content
    return raw_text

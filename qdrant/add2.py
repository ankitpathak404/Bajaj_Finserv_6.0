import os
from pdfminer.high_level import extract_text
from docx import Document
import email
from email import policy
from email.parser import BytesParser
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
import pandas as pd
from PIL import Image
import pytesseract
# Set the tesseract executable path (adjust path as needed)
pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'

# Load the model

model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B",device="cuda")


client = QdrantClient(
    url="https://7ce5a2a4-bd12-4594-bf67-3add19a2a39b.us-west-2-0.aws.cloud.qdrant.io:6333", 
    api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.tVu-4QA6NR3Mv5AnRV-b5Rlz1QGz7tYqi9s2PRffLrA",
)
def extract_pdf_text(file_path):
    """
    Extracts text from a PDF file using pdfminer.six
    """
    try:
        return extract_text(file_path)
    except Exception as e:
        print(f"[PDF ERROR] {file_path}: {e}")
        return ""

def extract_docx_text(file_path):
    """
    Extracts text from a DOCX file using python-docx
    """
    try:
        doc = Document(file_path)
        return "\n".join([para.text for para in doc.paragraphs])
    except Exception as e:
        print(f"[DOCX ERROR] {file_path}: {e}")
        return ""

def extract_email_text(file_path):
    """
    Extracts text from a .eml email file using Python's email parser
    """
    try:
        with open(file_path, 'rb') as f:
            msg = BytesParser(policy=policy.default).parse(f)
        body = msg.get_body(preferencelist=('plain'))
        return body.get_content() if body else ""
    except Exception as e:
        print(f"[EMAIL ERROR] {file_path}: {e}")
        return ""

def extract_xlsx_text(file_path):
    """
    Extracts text from an XLSX file using pandas
    """
    try:
        # Read all sheets and combine their content
        xl_file = pd.ExcelFile(file_path)
        all_text = []
        for sheet_name in xl_file.sheet_names:
            df = pd.read_excel(file_path, sheet_name=sheet_name)
            # Convert dataframe to string representation
            sheet_text = f"Sheet: {sheet_name}\n" + df.to_string(index=False)
            all_text.append(sheet_text)
        return "\n\n".join(all_text)
    except Exception as e:
        print(f"[XLSX ERROR] {file_path}: {e}")
        return ""

def extract_image_text(file_path):
    """
    Extracts text from image files (JPEG, PNG) using OCR (pytesseract)
    """
    try:
        image = Image.open(file_path)
        text = pytesseract.image_to_string(image)
        return text
    except Exception as e:
        print(f"[IMAGE ERROR] {file_path}: {e}")
        return ""

#payload={"document_name":"extracted_text"}
def extract_text_from_file(file_path):
    """
    General handler based on file type
    """
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        return extract_pdf_text(file_path)
    elif ext == ".docx":
        return extract_docx_text(file_path)
    elif ext == ".eml":
        return extract_email_text(file_path)
    elif ext == ".xlsx":
        return extract_xlsx_text(file_path)
    elif ext in [".jpeg", ".jpg", ".png", ".bmp", ".tiff", ".tif"]:
        return extract_image_text(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")

from langchain.text_splitter import RecursiveCharacterTextSplitter

def langchain_chunk(text):
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=400,           # adjust as needed (1000 for vector DBs)
        chunk_overlap=100,
        model_name="gpt-3.5-turbo",
        separators=["\n\n", "\n", ".", " ", ""]
    )
    return splitter.split_text(text)


if __name__ == "__main__":
  
    
    idx=4016


    print(f"\n=== Extracting:===")
    text = """9+5=22
            100+23=10023
            22+100=2200
            65007+2=650072"""
    text=langchain_chunk(text)
        
    for i in range(len(text)): 

            print(idx,"\n")
            v=model.encode(text[i])
            client.upsert(
                collection_name="bajaj",
                points=[
                 models.PointStruct(
            id=idx,
            payload={
                str(file): str(text[i]),
                "file_name":str(file)     
                     
                     },
            vector=v,  # ← just a list, not a dict
            )
            ])
            idx+=1

    text = "\n\n".join(text)  # Join chunks with double newlines for better readability
    # Save to .txt file
    txt_filename = os.path.splitext(file)[0] + ".txt"
    with open(txt_filename, "w", encoding="utf-8") as f:
        f.write(text)

    print(f"[SAVED] Extracted text saved to: {txt_filename}")
    print("\n--- End of Output ---\n")
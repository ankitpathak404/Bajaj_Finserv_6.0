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
from pptx import Presentation
from io import BytesIO
# Set the tesseract executable path (adjust path as needed)
pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'

# Load the model

model = SentenceTransformer("BAAI/bge-large-en-v1.5",device="cuda")


client = QdrantClient(
    url="https://7ce5a2a4-bd12-4594-bf67-3add19a2a39b.us-west-2-0.aws.cloud.qdrant.io:6333", 
    api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.tVu-4QA6NR3Mv5AnRV-b5Rlz1QGz7tYqi9s2PRffLrA",
)



if __name__ == "__main__":
    # Example usage
    
    idx=0
    
    string="""How does Newton derive Kepler's Second Law (equal areas in equal times) from his laws of motion and gravitation?
"""
    text=model.encode(string)
    response=client.query_points(
    collection_name="bajaj",
    query=text,
    with_payload=True,
    )

    print(response)
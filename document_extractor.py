import os
from pdfminer.high_level import extract_text
from docx import Document
import email
from email import policy
from email.parser import BytesParser
from qdrant_client import QdrantClient, models
from huggingface_hub import InferenceClient
import req
import torch
import tempfile
from dotenv import load_dotenv
import os

# Load variables from .env file
load_dotenv()
print(torch.cuda.is_available())  # Should return True



from sentence_transformers import SentenceTransformer

model = SentenceTransformer("BAAI/bge-large-en-v1.5",device="cuda")

def encode(text):
    return model.encode(text)



def download_file_from_url(url, save_dir="downloads"):
    os.makedirs(save_dir, exist_ok=True)  # Ensure destination exists
    filename = url.split("?")[0].split("/")[-1]  # Get clean filename
    response = req.get(url)
    if response.status_code != 200:
        raise Exception(f"Failed to download file: {url}")

    local_path = os.path.join(save_dir, filename)
    with open(local_path, "wb") as f:
        f.write(response.content)
    return local_path

client = QdrantClient(
    url="https://7ce5a2a4-bd12-4594-bf67-3add19a2a39b.us-west-2-0.aws.cloud.qdrant.io:6333", 
    api_key=os.getenv("api_key"),
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
    else:
        raise ValueError(f"Unsupported file type: {ext}")

from langchain.text_splitter import RecursiveCharacterTextSplitter

def langchain_chunk(text):
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=500,           # adjust as needed (1000 for vector DBs)
        chunk_overlap=0,
        model_name="gpt-3.5-turbo",
        separators=["\n\n", "\n", ".", " ", ""]
    )
    return splitter.split_text(text)


if __name__ == "__main__":
    # Example usage
    test_links = [
    "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D"
    ]

    # Download the files first
    test_files = []
    for url in test_links:
        try:
            downloaded_path = download_file_from_url(url,save_dir=".")
            test_files.append(downloaded_path)
        except Exception as e:
            print(f"[ERROR] Could not download: {url} → {e}")

    # Declare a global counter before the loop
    global_id = 0

    for file in test_files:
        if not os.path.exists(file):
            print(f"[SKIP] File not found: {file}")
            continue

        print(f"\n=== Processing: {file} ===")
        
        text = extract_text_from_file(file)
        text_chunks = langchain_chunk(text)

        # Batch encode
        vectors = encode(text_chunks)

        # Build batched points with serial IDs
        points = []
        for i, chunk in enumerate(text_chunks):
            print(i,"\n")
            points.append(
                models.PointStruct(
                    id=global_id,
                    payload={str(file): chunk},
                    vector=vectors[i]
                )
            )
            print(f"Chunk {i} -> Qdrant ID: {global_id}")
            global_id += 1  # Increment global serial ID

        # Batch upload
        client.upsert(
            collection_name="bajaj",
            points=points
        )

        print(f"✅ Uploaded {len(points)} chunks from {file}")


        text=str(text)
        # Save to .txt file
        txt_filename = os.path.splitext(file)[0] + ".txt"
        with open(txt_filename, "w", encoding="utf-8") as f:
            f.write(text)

        print(f"[SAVED] Extracted text saved to: {txt_filename}")
        print("\n--- End of Output ---\n")


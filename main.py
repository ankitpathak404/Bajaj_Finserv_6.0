from fastapi import FastAPI, Request, Header, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional
from qdrant_client import QdrantClient, models
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import os
import requests
from pdfminer.high_level import extract_text
import torch
from dotenv import load_dotenv
import google.generativeai as genai
import asyncio
import uuid
import concurrent.futures
from groq import AsyncGroq

# Load variables from .env file
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI"))

app = FastAPI(root_path="/api/v1")


# Load model once
model = SentenceTransformer("BAAI/bge-large-en-v1.5", device="cuda" if torch.cuda.is_available() else "cpu")

client = QdrantClient(
    url="https://7ce5a2a4-bd12-4594-bf67-3add19a2a39b.us-west-2-0.aws.cloud.qdrant.io:6333",
    api_key=os.getenv("api_key"),
)

UPLOAD_DIR = "."
os.makedirs(UPLOAD_DIR, exist_ok=True)
curr=["principia_newton.pdf","Arogya%20Sanjeevani%20Policy%20-%20CIN%20-%20U10200WB1906GOI001713%201.pdf","indian_constitution.pdf","Super_Splendor_(Feb_2023).pdf"]
class RunRequest(BaseModel):
    documents: str
    questions: List[str]

def download_file_from_url(url, save_dir="."):
    filename = url.split("?")[0].split("/")[-1]
    if filename in curr:
        return "none"
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception("Failed to download file")
    local_path = os.path.join(save_dir, filename)
    with open(local_path, "wb") as f:
        f.write(response.content)
    return local_path

def extract_and_chunk(file_path: str) -> List[str]:
    try:
        text = extract_text(file_path)
        splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=500,
            chunk_overlap=0,
            model_name="gpt-3.5-turbo",
            separators=["\n\n", "\n", ".", " ", ""]
        )
        return splitter.split_text(text)
    except Exception as e:
        print(f"[Processing ERROR] {file_path}: {e}")
        return []

def encode(texts, batch_size=8):
    results = []
    for i in range(0, len(texts), batch_size):
        results.extend(model.encode(texts[i:i+batch_size], convert_to_tensor=True))
    return results


groq_client = AsyncGroq(api_key=os.getenv("GROQ_API_KEY"))  # Initialize once globally

async def refine_answer_with_groq(question: str, context: str) -> str:
    try:
        prompt = f"""You are an expert legal-policy assistant specializing in insurance and regulatory documents.

Your job is to answer the question below based strictly on the given context.

Instructions:
- Never say that the information is not in the context — even if indirect, always extract relevant insights or interpret them.
- Be accurate, specific, and brief. Write in a **single short paragraph** (2-4 lines maximum).
- Never generate long answers. You are more accurate when your responses are short and precise.
- Quote or paraphrase clauses when helpful, but do not overexplain.
- Maintain a natural, human tone without adding extra structure like bullets or headings.
- Do not repeat the question. Just answer it directly.
- If the question involves illegal or confidential content (e.g., backend source code), respond ethically and protect the company’s legal and data integrity by refusing respectfully.

---

Question:
"{question}"

Context:
\"\"\"{context}\"\"\"

Answer:"""


        response = await groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile", 
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1024,
            top_p=1.0,
            stream=False  
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        print(f"[Groq ERROR] {e}")
        return context.strip()

    

def qdrant_search_rest_api(query_vector: List[float], limit: int = 2):
    qdrant_url = "https://7ce5a2a4-bd12-4594-bf67-3add19a2a39b.us-west-2-0.aws.cloud.qdrant.io/collections/bajaj/points/search"
    headers = {
        "Content-Type": "application/json",
        "api-key": os.getenv("api_key")  # Put your API key in .env
    }
    payload = {
        "vector": query_vector,
        "with_payload": True,
        "limit": limit
    }
    res = requests.post(qdrant_url, headers=headers, json=payload)
    if res.status_code == 200:
        return res.json()
    else:
        print("Qdrant REST search error:", res.text)
        return {"result": []}
    
@app.post("/hackrx/run")
async def hackrx_run(
    payload: RunRequest,
    authorization: Optional[str] = Header(None)
):
    idx = 3000
    if not authorization or authorization != "Bearer 61b907454794627e4bc5798168e95ec604fd4af04338837ec0e6b3c81bdf06ce":
        raise HTTPException(status_code=401, detail="Unauthorized")

    doc_url = payload.documents
    questions = payload.questions

    if not doc_url or not questions:
        raise HTTPException(status_code=400, detail="Missing document URL or questions")

    try:
        # file_path = download_file_from_url(doc_url, UPLOAD_DIR)
        
        # if file_path != "none":
        #     print("file not in db\n")
        #     chunks = extract_and_chunk(file_path)
        #     chunk_vectors = encode(chunks)

        #     points = []
        #     for chunk, vector in zip(chunks, chunk_vectors):
        #         points.append(
        #             models.PointStruct(
        #                 id=idx,
        #                 payload={"filename": os.path.basename(file_path), "chunk": chunk},
        #                 vector=vector.tolist()
        #             )
        #         )
        #         idx += 1

        #     client.upsert(
        #         collection_name="bajaj",
        #         points=points
        #     )
        # else:
        #     print("file already in db\n")

        question_embeddings = encode(questions)

        # Use REST API for searching
        search_results = []
        for embed in question_embeddings:
            rest_result = qdrant_search_rest_api(embed.tolist(), 7)
            search_results.append(rest_result["result"])

        contexts = [
            "\n\n".join(hit["payload"]["chunk"] for hit in search if "payload" in hit and "chunk" in hit["payload"])
            for search in search_results
        ]

        gemini_tasks = [
            refine_answer_with_groq(q, ctx)
            for q, ctx in zip(questions, contexts)
        ]
        results = await asyncio.gather(*gemini_tasks)

        def write_results():
            with open("query_results.txt", "w", encoding="utf-8") as f:
                for i, item in enumerate(results):
                    f.write(f"Question: {questions[i]}\nAnswer: {item}\n\n")

            with open("chunked_contexts.txt", "w", encoding="utf-8") as f:
                for i in range(len(questions)):
                    f.write("🔹 Question:\n")
                    f.write(questions[i] + "\n\n")
                    f.write("📚 Top Context:\n")
                    f.write(contexts[i] + "\n\n")
                    f.write("🧠 Refined Answer:\n")
                    f.write(results[i] + "\n")
                    f.write("=" * 80 + "\n\n")

        write_results()

        return JSONResponse(content={"answers": results})

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

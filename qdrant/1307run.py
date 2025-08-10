from fastapi import FastAPI, Request, Header, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional
import httpx
import asyncio
import requests
import torch
from dotenv import load_dotenv
import os
from groq import AsyncGroq
from openai import OpenAI
from urllib.parse import unquote
import difflib
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
import time
from datetime import datetime
import json
import fitz  # PyMuPDF for PDF extraction
import docx  # python-docx for DOCX extraction
import pandas as pd  # for XLSX extraction
import re
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load variables from .env file
load_dotenv()

app = FastAPI(root_path="/api/v1")

groq_client = AsyncGroq(api_key=os.getenv("GROQ_API_KEY"))
client = QdrantClient(
    url="https://7ce5a2a4-bd12-4594-bf67-3add19a2a39b.us-west-2-0.aws.cloud.qdrant.io:6333",
    api_key=os.getenv("api_key"),
)
from sentence_transformers import SentenceTransformer
import os
from huggingface_hub import login

login(token=os.getenv("HF_TOKEN"))
model = SentenceTransformer("BAAI/bge-large-en-v1.5", device="cuda" if torch.cuda.is_available() else "cpu")

# ✅ Set your vector store ID here
vector_store_id = "vs_688e116ab5b08191a3c1561d9c3d78ec"

UPLOAD_DIR = "."
os.makedirs(UPLOAD_DIR, exist_ok=True)
global_id = 0

# ✅ NEW: City to Landmark mapping from the PDF
CITY_TO_LANDMARK = {
    "Delhi": "Gateway of India",
    "Mumbai": "India Gate", 
    "Chennai": "Charminar",
    "Hyderabad": "Taj Mahal",
    "Ahmedabad": "Howrah Bridge",
    "Kolkata": "Taj Mahal",
    "Bengaluru": "Eiffel Tower",
    "Pune": "Big Ben",
    "Ahmedabad": "Howrah Bridge",
    "Jaipur": "Sydney Opera House",
    "Lucknow": "Christ the Redeemer"
    # Add more mappings as needed from the PDF
}

# ✅ NEW: Landmark to flight endpoint mapping
LANDMARK_TO_FLIGHT_ENDPOINT = {
    "Gateway of India": "https://register.hackrx.in/teams/public/flights/getFirstCityFlightNumber",
    "Taj Mahal": "https://register.hackrx.in/teams/public/flights/getSecondCityFlightNumber", 
    "Eiffel Tower": "https://register.hackrx.in/teams/public/flights/getThirdCityFlightNumber",
    "Big Ben": "https://register.hackrx.in/teams/public/flights/getFourthCityFlightNumber"
}

DEFAULT_FLIGHT_ENDPOINT = "https://register.hackrx.in/teams/public/flights/getFifthCityFlightNumber"

class RunRequest(BaseModel):
    documents: str
    questions: List[str]

class FavouriteCityRequest(BaseModel):
    questions: List[str]

# ✅ NEW: Enhanced request model for chunk extraction
class ChunkExtractionRequest(BaseModel):
    documents: str
    questions: List[str]
    extract_chunks: Optional[bool] = False  # Optional flag to enable chunk extraction

from qdrant_client.http.models import PayloadSchemaType

try:
    client.create_payload_index(
        collection_name="bajaj",
        field_name="file_name",
        field_schema=PayloadSchemaType.KEYWORD  # This makes "file_name" filterable
    )
except Exception as e:
    logger.info(f"Payload index might already exist: {e}")

# ✅ NEW: Function to detect if URL is an API endpoint that returns tokens/data
def is_api_endpoint(url: str) -> bool:
    """Check if URL is likely an API endpoint that returns JSON data"""
    api_indicators = [
        'get-secret-token',
        '/api/',
        'token',
        'secret',
        '/utils/',
        'hackTeam='
    ]
    return any(indicator in url.lower() for indicator in api_indicators)

# ✅ NEW: Function to fetch content from API endpoints
async def fetch_api_content(url: str) -> tuple:
    """
    Fetch content from API endpoint
    Returns: (content_text, is_token, raw_response)
    """
    try:
        async with httpx.AsyncClient(timeout=30.0) as http_client:
            response = await http_client.get(url)
            response.raise_for_status()
            
            # Try to parse as JSON
            try:
                json_data = response.json()
                
                # Check if response contains a token
                token_keys = ['token', 'secret_token', 'secretToken', 'secret', 'key']
                for key in token_keys:
                    if key in json_data:
                        token_value = json_data[key]
                        logger.info(f"Found token in response: {key}={token_value}")
                        return str(token_value), True, json_data
                
                # If no specific token key, return the entire JSON as text
                json_str = json.dumps(json_data, indent=2)
                logger.info(f"API returned JSON data: {json_str[:200]}...")
                return json_str, False, json_data
                
            except json.JSONDecodeError:
                # If not JSON, return as plain text
                text_content = response.text
                logger.info(f"API returned text content: {text_content[:200]}...")
                return text_content, False, text_content
                
    except Exception as e:
        logger.error(f"Failed to fetch API content from {url}: {e}")
        return f"Error fetching content: {str(e)}", False, None

# ✅ NEW: Enhanced function to handle both file downloads and API calls
# --- process_document_url.py ---
async def process_document_url(url: str) -> tuple:
    """
    Process URL - either download file or fetch API content
    Returns: (content, file_name, is_api_content, raw_response)
    """
    if is_api_endpoint(url):
        logger.info(f"Detected API endpoint: {url}")
        content, is_token, raw_response = await fetch_api_content(url)

        # Always save API response to disk so refine() can open it
        filename = f"api_response_{int(time.time())}.json"
        with open(filename, "w", encoding="utf-8") as f:
            # Save in a predictable key for refine()
            if isinstance(raw_response, (dict, list)):
                json.dump({"raw": raw_response, "html": content}, f, ensure_ascii=False, indent=2)
            else:
                json.dump({"html": str(content)}, f, ensure_ascii=False, indent=2)

        logger.info(f"Saved API content to {filename}")
        return content, filename, True, raw_response

    else:
        # Regular file processing
        filename = download_file_from_url(url, UPLOAD_DIR)
        return None, filename, False, None



# ✅ NEW: Flight number retrieval workflow
async def get_flight_number() -> tuple:
    """
    Multi-step workflow to get flight number:
    1. Fetch favourite city
    2. Map city to landmark
    3. Choose correct flight path
    4. Return flight number with process details
    Returns: (flight_number, process_details)
    """
    process_details = {
        "step1_city_fetch": {},
        "step2_landmark_mapping": {},
        "step3_endpoint_selection": {},
        "step4_flight_fetch": {},
        "summary": {}
    }
    
    try:
        # Step 1: Fetch favourite city
        logger.info("Step 1: Fetching favourite city...")
        process_details["step1_city_fetch"]["status"] = "in_progress"
        process_details["step1_city_fetch"]["api_url"] = "https://register.hackrx.in/submissions/myFavouriteCity"
        
        async with httpx.AsyncClient(timeout=5.0) as http_client:
            response = await http_client.get("https://register.hackrx.in/submissions/myFavouriteCity")
            response.raise_for_status()
            
            city_data = response.json()
            process_details["step1_city_fetch"]["raw_response"] = city_data
            process_details["step1_city_fetch"]["http_status"] = response.status_code
            logger.info(f"City data received: {city_data}")
            
            # Extract city from response
            if "data" in city_data and "city" in city_data["data"]:
                city = city_data["data"]["city"]
                process_details["step1_city_fetch"]["extracted_city"] = city
                process_details["step1_city_fetch"]["status"] = "success"
                logger.info(f"Retrieved city: {city}")
            else:
                process_details["step1_city_fetch"]["status"] = "failed"
                process_details["step1_city_fetch"]["error"] = "Invalid city data format"
                raise HTTPException(status_code=502, detail="Invalid city data format")
        
        # Step 2: Map city to landmark
        logger.info("Step 2: Mapping city to landmark...")
        process_details["step2_landmark_mapping"]["input_city"] = city
        process_details["step2_landmark_mapping"]["mapping_table_used"] = CITY_TO_LANDMARK
        
        landmark = CITY_TO_LANDMARK.get(city, "Unknown")
        process_details["step2_landmark_mapping"]["mapped_landmark"] = landmark
        process_details["step2_landmark_mapping"]["mapping_found"] = landmark != "Unknown"
        logger.info(f"City '{city}' mapped to landmark: '{landmark}'")
        
        # Step 3: Choose correct flight path
        logger.info("Step 3: Choosing flight endpoint...")
        process_details["step3_endpoint_selection"]["input_landmark"] = landmark
        process_details["step3_endpoint_selection"]["available_endpoints"] = LANDMARK_TO_FLIGHT_ENDPOINT
        process_details["step3_endpoint_selection"]["default_endpoint"] = DEFAULT_FLIGHT_ENDPOINT
        
        flight_endpoint = LANDMARK_TO_FLIGHT_ENDPOINT.get(landmark, DEFAULT_FLIGHT_ENDPOINT)
        process_details["step3_endpoint_selection"]["selected_endpoint"] = flight_endpoint
        process_details["step3_endpoint_selection"]["used_default"] = landmark not in LANDMARK_TO_FLIGHT_ENDPOINT
        logger.info(f"Using flight endpoint: {flight_endpoint}")
        
        # Step 4: Get flight number
        logger.info("Step 4: Fetching flight number...")
        process_details["step4_flight_fetch"]["endpoint_called"] = flight_endpoint
        process_details["step4_flight_fetch"]["status"] = "in_progress"
        
        async with httpx.AsyncClient(timeout=5.0) as http_client:
            flight_response = await http_client.get(flight_endpoint)
            flight_response.raise_for_status()
            
            flight_data = flight_response.json()
            process_details["step4_flight_fetch"]["raw_response"] = flight_data
            process_details["step4_flight_fetch"]["http_status"] = flight_response.status_code
            logger.info(f"Flight data received: {flight_data}")
            
            # Extract flight number
            if "data" in flight_data and "flightNumber" in flight_data["data"]:
                flight_number = flight_data["data"]["flightNumber"]
                process_details["step4_flight_fetch"]["extracted_flight_number"] = flight_number
                process_details["step4_flight_fetch"]["status"] = "success"
                logger.info(f"Final flight number: {flight_number}")
                
                # Summary
                process_details["summary"] = {
                    "workflow_status": "completed_successfully",
                    "city_retrieved": city,
                    "landmark_mapped": landmark,
                    "endpoint_used": flight_endpoint,
                    "final_flight_number": flight_number,
                    "all_steps_successful": True
                }
                
                return flight_number, process_details
            else:
                process_details["step4_flight_fetch"]["status"] = "failed"
                process_details["step4_flight_fetch"]["error"] = "Invalid flight data format"
                raise HTTPException(status_code=502, detail="Invalid flight data format")
                
    except httpx.HTTPStatusError as e:
        error_info = {
            "error_type": "HTTP_ERROR",
            "status_code": e.response.status_code,
            "response_text": e.response.text,
            "failed_step": "API_CALL"
        }
        process_details["summary"] = {
            "workflow_status": "failed",
            "error_details": error_info,
            "all_steps_successful": False
        }
        logger.error(f"HTTP error in flight workflow: {e.response.status_code} - {e.response.text}")
        raise HTTPException(status_code=502, detail=f"Flight workflow HTTP error: {e.response.status_code}")
    except Exception as e:
        error_info = {
            "error_type": type(e).__name__,
            "error_message": str(e),
            "failed_step": "GENERAL_ERROR"
        }
        process_details["summary"] = {
            "workflow_status": "failed",
            "error_details": error_info,
            "all_steps_successful": False
        }
        logger.error(f"Error in flight workflow: {e}")
        raise HTTPException(status_code=500, detail=f"Flight workflow error: {str(e)}")

# ✅ NEW: Fetch favourite city from the API
async def fetch_favourite_city() -> dict:
    """Fetch favourite city from the specified API endpoint"""
    try:
        async with httpx.AsyncClient(timeout=30.0) as http_client:
            response = await http_client.get("https://register.hackrx.in/submissions/myFavouriteCity")
            response.raise_for_status()
            
            city_data = response.json()
            logger.info(f"Fetched favourite city data: {city_data}")
            return city_data
            
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error fetching favourite city: {e.response.status_code} - {e.response.text}")
        raise HTTPException(status_code=502, detail=f"Failed to fetch favourite city: HTTP {e.response.status_code}")
    except Exception as e:
        logger.error(f"Error fetching favourite city: {e}")
        raise HTTPException(status_code=502, detail=f"Failed to fetch favourite city: {str(e)}")

# ✅ NEW: Process questions using favourite city document
async def process_favourite_city_questions(questions: List[str], city_data: dict) -> List[str]:
    """Process questions using the favourite city document URL"""
    try:
        # Extract document URL from city data
        if "document" not in city_data:
            raise HTTPException(status_code=400, detail="No document URL found in favourite city response")
        
        doc_url = city_data["document"]
        
        # Extract filename from URL
        file_path = download_file_from_url(doc_url, UPLOAD_DIR)
        file_name = unquote(file_path)
        logger.info(f"Processing favourite city document: {file_name}")
        
        # Ensure document is in Qdrant
        ingestion_success = await ensure_document_in_qdrant(doc_url, file_name)
        
        if not ingestion_success:
            logger.warning(f"Failed to ensure document {file_name} is in Qdrant, proceeding with search anyway")
        
        # Process questions in parallel
        tasks = [refine(q, file_name) for q in questions]
        responses = await asyncio.gather(*tasks)
        
        # Extract answers
        answers = [response[0] for response in responses]
        
        logger.info(f"Generated {len(answers)} answers for favourite city questions")
        return answers
        
    except Exception as e:
        logger.error(f"Error processing favourite city questions: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process questions: {str(e)}")

def create_log_entry(request_start_time, doc_url, questions, results, file_name, ingestion_status, total_time, all_chunks=None, additional_info=None):
    """Create comprehensive log entry"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # Include milliseconds
    
    log_data = {
        "timestamp": datetime.now().isoformat(),
        "request_start_time": request_start_time,
        "total_processing_time_seconds": total_time,
        "document_received": {
            "original_url": doc_url,
            "extracted_filename": file_name,
            "ingestion_status": ingestion_status,
            "dynamic_ingestion_used": True
        },
        "questions_received": questions,
        "total_questions": len(questions),
        "responses": results,
        "retrieved_chunks": all_chunks if all_chunks else [],
        "api_stats": {
            "groq_model_used": "llama-3.1-8b-instant",
            "vector_search_performed": True,
            "qdrant_collection": "bajaj"
        },
        "system_info": {
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "sentence_transformer_model": "BAAI/bge-large-en-v1.5"
        }
    }
    
    if additional_info:
        log_data.update(additional_info)
    
    # Create log file with timestamp
    readable_name = datetime.now().strftime("%d-%m-%Y_%I-%M-%S%p")
    log_filename = f"API_Request_{readable_name}.txt"
    
    with open(log_filename, "w", encoding="utf-8") as f:
        f.write("="*80 + "\n")
        f.write(f"API REQUEST LOG - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"REQUEST TIMESTAMP: {log_data['timestamp']}\n")
        f.write(f"TOTAL PROCESSING TIME: {total_time:.3f} seconds\n\n")
        
        f.write("DOCUMENT INFORMATION:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Original URL: {doc_url}\n")
        f.write(f"Extracted Filename: {file_name}\n")
        f.write(f"Ingestion Status: {ingestion_status}\n")
        f.write(f"Dynamic Ingestion: {'✓ ENABLED' if log_data['document_received']['dynamic_ingestion_used'] else '✗ DISABLED'}\n\n")
        
        f.write(f"QUESTIONS RECEIVED ({len(questions)} total):\n")
        f.write("-" * 30 + "\n")
        for i, q in enumerate(questions, 1):
            f.write(f"{i}. {q}\n")
        f.write("\n")
        
        f.write("RESPONSES GENERATED:\n")
        f.write("-" * 30 + "\n")
        for i, result in enumerate(results, 1):
            f.write(f"Q{i}: {result['question']}\n")
            f.write(f"A{i}: {result['answer']}\n")
            f.write("-" * 50 + "\n")
        
        if all_chunks:
            f.write("\nRETRIEVED CHUNKS BY QUESTION:\n")
            f.write("="*60 + "\n")
            for i, question_chunks in enumerate(all_chunks, 1):
                f.write(f"\nQUESTION {i}: {questions[i-1]}\n")
                f.write("-" * 50 + "\n")
                for j, chunk in enumerate(question_chunks, 1):
                    f.write(f"Chunk {j} (Score: {chunk['score']:.4f}):\n")
                    f.write(f"File: {chunk['file_name']}\n")
                    f.write(f"Content: {chunk['content']}\n")
                    f.write("-" * 40 + "\n")
        
        f.write("\nSYSTEM INFORMATION:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Device: {log_data['system_info']['device']}\n")
        f.write(f"Model: {log_data['system_info']['sentence_transformer_model']}\n")
        f.write(f"Groq Model: {log_data['api_stats']['groq_model_used']}\n")
        f.write(f"Vector DB: {log_data['api_stats']['qdrant_collection']}\n\n")
        
        f.write("RAW JSON DATA:\n")
        f.write("-" * 30 + "\n")
        f.write(json.dumps(log_data, indent=2, ensure_ascii=False))
        f.write("\n\n" + "="*80 + "\n")
    
    logger.info(f"📝 Log saved to: {log_filename}")
    return log_filename

# ✅ NEW: Enhanced log entry for chunk extraction
def create_chunk_extraction_log(request_start_time, doc_url, questions, results, file_name, processing_stats, total_time, all_chunks=None):
    """Create detailed log entry for chunk extraction workflow"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    
    log_data = {
        "timestamp": datetime.now().isoformat(),
        "workflow_type": "chunk_extraction_enabled",
        "request_start_time": request_start_time,
        "total_processing_time_seconds": total_time,
        "document_info": {
            "original_url": doc_url,
            "extracted_filename": file_name,
            "text_content_length": processing_stats.get("text_content_length", 0)
        },
        "chunk_extraction_stats": processing_stats.get("relevant_chunks", {}),
        "upload_stats": processing_stats.get("upload_stats", {}),
        "questions_processed": questions,
        "responses": results,
        "processing_success": {
            "document_processed": processing_stats.get("document_processed", False),
            "chunks_extracted": processing_stats.get("chunks_extracted", False),
            "chunks_uploaded": processing_stats.get("chunks_uploaded", False)
        }
    }
    
    readable_name = datetime.now().strftime("%d-%m-%Y_%I-%M-%S%p")
    log_filename = f"ChunkExtraction_Log_{readable_name}.txt"
    
    with open(log_filename, "w", encoding="utf-8") as f:
        f.write("="*80 + "\n")
        f.write(f"CHUNK EXTRACTION WORKFLOW LOG - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"REQUEST TIMESTAMP: {log_data['timestamp']}\n")
        f.write(f"TOTAL PROCESSING TIME: {total_time:.3f} seconds\n")
        f.write(f"WORKFLOW TYPE: {log_data['workflow_type']}\n\n")
        
        f.write("DOCUMENT PROCESSING STATUS:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Document Processed: {'✓' if processing_stats.get('document_processed') else '✗'}\n")
        f.write(f"Chunks Extracted: {'✓' if processing_stats.get('chunks_extracted') else '✗'}\n")
        f.write(f"Chunks Uploaded: {'✓' if processing_stats.get('chunks_uploaded') else '✗'}\n")
        f.write(f"Text Content Length: {processing_stats.get('text_content_length', 0)} characters\n\n")
        
        f.write("UPLOAD STATISTICS:\n")
        f.write("-" * 30 + "\n")
        upload_stats = processing_stats.get("upload_stats", {})
        f.write(f"Total Questions: {upload_stats.get('total_questions', 0)}\n")
        f.write(f"Total Chunks Uploaded: {upload_stats.get('total_chunks_uploaded', 0)}\n")
        f.write(f"Upload Success: {'✓' if upload_stats.get('upload_success') else '✗'}\n\n")
        
        f.write("RELEVANT CHUNKS BY QUESTION:\n")
        f.write("="*60 + "\n")
        relevant_chunks = processing_stats.get("relevant_chunks", {})
        for i, (question, chunks) in enumerate(relevant_chunks.items(), 1):
            f.write(f"\nQUESTION {i}: {question}\n")
            f.write("-" * 50 + "\n")
            f.write(f"Relevant chunks found: {len(chunks)}\n")
            
            for j, chunk_info in enumerate(chunks, 1):
                f.write(f"\nChunk {j}:\n")
                f.write(f"  Similarity Score: {chunk_info.get('similarity_score', 0):.4f}\n")
                f.write(f"  Chunk Index: {chunk_info.get('chunk_index', 'N/A')}\n")
                f.write(f"  Content Length: {chunk_info.get('content_length', 0)} chars\n")
                f.write(f"  Content Preview: {chunk_info.get('content', '')[:200]}...\n")
                f.write("-" * 40 + "\n")
        
        f.write("\nQUESTIONS & ANSWERS:\n")
        f.write("-" * 30 + "\n")
        for i, result in enumerate(results, 1):
            f.write(f"Q{i}: {result['question']}\n")
            f.write(f"A{i}: {result['answer']}\n")
            f.write("-" * 50 + "\n")
        
        f.write("\nRAW JSON DATA:\n")
        f.write("-" * 30 + "\n")
        f.write(json.dumps(log_data, indent=2, ensure_ascii=False))
        f.write("\n\n" + "="*80 + "\n")
    
    logger.info(f"📝 Chunk extraction log saved to: {log_filename}")
    return log_filename

@app.post("/hackrx/run")
async def hackrx_run(
    payload: RunRequest,
    authorization: Optional[str] = Header(None)
):
    global global_id
    request_start_time = time.time()
    start_timestamp = datetime.now().isoformat()

    if not authorization or authorization != "Bearer 61b907454794627e4bc5798168e95ec604fd4af04338837ec0e6b3c81bdf06ce":
        raise HTTPException(status_code=401, detail="Unauthorized")

    doc_url = payload.documents
    questions = payload.questions

    if not doc_url or not questions:
        raise HTTPException(status_code=400, detail="Missing document URL or questions")

    try:
        # ✅ NEW: Enhanced URL processing to handle API endpoints
        api_content, file_name, is_api_content, raw_response = await process_document_url(doc_url)
        file_name = unquote(file_name)
        logger.info(f"Processing {'API endpoint' if is_api_content else 'document'}: {file_name}")

        if is_api_content:
            # Handle API content directly
            logger.info("Processing API endpoint response")
            
            # Check if the API response contains a token
            is_token = False
            if raw_response:
                token_keys = ['token', 'secret_token', 'secretToken', 'secret', 'key']
                is_token = any(key in str(raw_response).lower() for key in token_keys)
            
            # Process questions using API content
            tasks = [refine(q, file_name) for q in questions]
            responses = await asyncio.gather(*tasks)
            
            ingestion_status = "api_content_processed"
        else:
            # Regular document processing
            # Ensure document is in Qdrant before proceeding
            ingestion_success = await ensure_document_in_qdrant(doc_url, file_name)
            ingestion_status = "success" if ingestion_success else "failed"
            
            if not ingestion_success:
                logger.warning(f"Failed to ensure document {file_name} is in Qdrant, proceeding with search anyway")
            
            # Question Answering - Kick off all refine() calls in parallel
            tasks = [refine(q, file_name) for q in questions]
            responses = await asyncio.gather(*tasks)

        # Extract answers and chunks
        answers = [response[0] for response in responses]
        all_chunks = [response[1] for response in responses]

        # Zip the results
        results = [
            {"question": question, "answer": answer}
            for question, answer in zip(questions, answers)
        ]

        # Calculate total processing time
        total_time = time.time() - request_start_time

        # Save results to query_results.txt
        with open("query_results.txt", "w", encoding="utf-8") as f:
            for item in results:
                f.write(f"Question: {item['question']}\nAnswer: {item['answer']}\n\n")

        # Create comprehensive log after successful completion
        log_filename = create_log_entry(
            start_timestamp, 
            doc_url, 
            questions, 
            results, 
            file_name, 
            ingestion_status,
            total_time,
            all_chunks,
            additional_info={
                "status": "success",
                "answers_generated": len(answers),
                "parallel_processing": True,
                "is_api_content": is_api_content,
                "ingestion_performed": not is_api_content
            }
        )

        return JSONResponse(content={"answers": answers})

    except Exception as e:
        # Calculate time even for errors
        total_time = time.time() - request_start_time
        
        # Log error case
        error_log = create_log_entry(
            start_timestamp,
            doc_url,
            questions,
            [],
            file_name if 'file_name' in locals() else "unknown",
            "error",
            total_time,
            None,
            additional_info={
                "status": "error",
                "error_message": str(e),
                "error_type": type(e).__name__
            }
        )
        
        logger.error(f"Error processing request: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

# ✅ NEW ENHANCED ENDPOINT: Run with chunk extraction option
@app.post("/hackrx/run-with-chunks")
async def hackrx_run_with_chunks(
    payload: ChunkExtractionRequest,
    authorization: Optional[str] = Header(None)
):
    """
    Enhanced endpoint that supports chunk extraction workflow
    If extract_chunks=true, extracts relevant chunks and uploads only those to Qdrant
    """
    global global_id
    request_start_time = time.time()
    start_timestamp = datetime.now().isoformat()

    if not authorization or authorization != "Bearer 61b907454794627e4bc5798168e95ec604fd4af04338837ec0e6b3c81bdf06ce":
        raise HTTPException(status_code=401, detail="Unauthorized")

    doc_url = payload.documents
    questions = payload.questions
    extract_chunks = payload.extract_chunks

    if not doc_url or not questions:
        raise HTTPException(status_code=400, detail="Missing document URL or questions")

    # Extract filename from URL
    file_path = download_file_from_url(doc_url, UPLOAD_DIR)
    file_name = unquote(file_path)
    logger.info(f"Processing document with chunk extraction: {file_name} (extract_chunks={extract_chunks})")

    try:
        # Process document with optional chunk extraction
        processing_stats = await process_document_with_chunk_extraction(
            doc_url, file_name, questions, extract_chunks
        )
        
        # Question Answering - Kick off all refine() calls in parallel
        tasks = [refine(q, file_name) for q in questions]
        responses = await asyncio.gather(*tasks)

        # Extract answers and chunks
        answers = [response[0] for response in responses]
        all_chunks = [response[1] for response in responses]

        # Zip the results
        results = [
            {"question": question, "answer": answer}
            for question, answer in zip(questions, answers)
        ]

        # Calculate total processing time
        total_time = time.time() - request_start_time

        # Save results to query_results.txt
        with open("query_results.txt", "w", encoding="utf-8") as f:
            for item in results:
                f.write(f"Question: {item['question']}\nAnswer: {item['answer']}\n\n")

        # Create specialized log for chunk extraction workflow
        if extract_chunks:
            log_filename = create_chunk_extraction_log(
                start_timestamp, doc_url, questions, results, file_name, processing_stats, total_time, all_chunks
            )
        else:
            # Use regular log for standard processing
            log_filename = create_log_entry(
                start_timestamp, doc_url, questions, results, file_name, 
                "success" if processing_stats.get("document_processed") else "failed",
                total_time, all_chunks,
                additional_info={
                    "status": "success",
                    "answers_generated": len(answers),
                    "chunk_extraction_enabled": extract_chunks,
                    "processing_stats": processing_stats
                }
            )

        # Prepare response with additional metadata
        response_data = {
            "answers": answers,
            "metadata": {
                "processing_time_seconds": total_time,
                "chunk_extraction_enabled": extract_chunks,
                "document_processed": processing_stats.get("document_processed", False),
                "chunks_extracted": processing_stats.get("chunks_extracted", False),
                "chunks_uploaded": processing_stats.get("chunks_uploaded", False)
            }
        }
        
        # Include chunk extraction details if enabled
        if extract_chunks and processing_stats.get("chunks_extracted"):
            response_data["chunk_extraction_summary"] = {
                "total_questions": len(questions),
                "total_chunks_uploaded": processing_stats.get("upload_stats", {}).get("total_chunks_uploaded", 0),
                "relevant_chunks_found": sum(len(chunks) for chunks in processing_stats.get("relevant_chunks", {}).values())
            }

        return JSONResponse(content=response_data)

    except Exception as e:
        # Calculate time even for errors
        total_time = time.time() - request_start_time
        
        # Log error case
        error_log = create_log_entry(
            start_timestamp,
            doc_url,
            questions,
            [],
            file_name if 'file_name' in locals() else "unknown",
            "error",
            total_time,
            None,
            additional_info={
                "status": "error",
                "error_message": str(e),
                "error_type": type(e).__name__,
                "chunk_extraction_enabled": extract_chunks
            }
        )
        
        logger.error(f"Error processing request with chunk extraction: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

# ✅ NEW ENDPOINT: Favourite City Workflow
@app.post("/hackrx/favourite-city")
async def hackrx_favourite_city(
    payload: FavouriteCityRequest,
    authorization: Optional[str] = Header(None)
):
    """
    Multi-step workflow:
    1. Fetch favourite city data from API
    2. Process questions using the city document
    3. Return answers
    """
    request_start_time = time.time()
    start_timestamp = datetime.now().isoformat()

    # Validate authorization
    if not authorization or authorization != "Bearer 61b907454794627e4bc5798168e95ec604fd4af04338837ec0e6b3c81bdf06ce":
        raise HTTPException(status_code=401, detail="Unauthorized")

    questions = payload.questions

    if not questions:
        raise HTTPException(status_code=400, detail="Missing questions")

    try:
        logger.info("Starting favourite city workflow...")
        
        # Step 1: Fetch favourite city data
        logger.info("Step 1: Fetching favourite city data...")
        city_data = await fetch_favourite_city()
        logger.info(f"City data received: {json.dumps(city_data, indent=2)}")
        
        # Step 2: Process questions using the city document
        logger.info("Step 2: Processing questions with city document...")
        answers = await process_favourite_city_questions(questions, city_data)
        
        # Calculate total processing time
        total_time = time.time() - request_start_time
        
        logger.info(f"Favourite city workflow completed in {total_time:.3f} seconds")
        
        # Prepare results in the same format as /hackrx/run
        results = [
            {"question": question, "answer": answer}
            for question, answer in zip(questions, answers)
        ]
        
        # Log the favourite city workflow
        with open("favourite_city_log.txt", "w", encoding="utf-8") as f:
            f.write(f"FAVOURITE CITY WORKFLOW LOG - {datetime.now().isoformat()}\n")
            f.write("="*60 + "\n")
            f.write(f"Processing Time: {total_time:.3f} seconds\n")
            f.write(f"City Data Retrieved: {json.dumps(city_data, indent=2)}\n")
            f.write(f"Questions Processed: {len(questions)}\n")
            f.write(f"Answers Generated: {len(answers)}\n\n")
            
            for i, (q, a) in enumerate(zip(questions, answers), 1):
                f.write(f"Q{i}: {q}\n")
                f.write(f"A{i}: {a}\n")
                f.write("-" * 40 + "\n")

        # Return in the same format as the original endpoint
        return JSONResponse(content={"answers": answers})

    except HTTPException:
        # Re-raise HTTP exceptions (like 502 from fetch_favourite_city)
        raise
    except Exception as e:
        total_time = time.time() - request_start_time
        
        logger.error(f"Error in favourite city workflow: {e}")
        
        # Log error
        with open("favourite_city_error_log.txt", "w", encoding="utf-8") as f:
            f.write(f"FAVOURITE CITY WORKFLOW ERROR - {datetime.now().isoformat()}\n")
            f.write("="*60 + "\n")
            f.write(f"Error: {str(e)}\n")
            f.write(f"Error Type: {type(e).__name__}\n")
            f.write(f"Processing Time: {total_time:.3f} seconds\n")
            f.write(f"Questions: {questions}\n")
        
        return JSONResponse(status_code=500, content={"error": str(e)})

# ✅ NEW ENDPOINT: Get favourite city data only
@app.get("/hackrx/city-info")
async def get_city_info(authorization: Optional[str] = Header(None)):
    """Get favourite city information"""
    
    if not authorization or authorization != "Bearer 61b907454794627e4bc5798168e95ec604fd4af04338837ec0e6b3c81bdf06ce":
        raise HTTPException(status_code=401, detail="Unauthorized")
    
    try:
        city_data = await fetch_favourite_city()
        return JSONResponse(content=city_data)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching city info: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

# ✅ NEW ENDPOINT: Direct flight number retrieval
@app.post("/hackrx/flight")
async def hackrx_flight(authorization: Optional[str] = Header(None)):
    """
    Direct flight number retrieval endpoint
    Follows the multi-step workflow to get flight number with detailed process info
    """
    if not authorization or authorization != "Bearer 61b907454794627e4bc5798168e95ec604fd4af04338837ec0e6b3c81bdf06ce":
        raise HTTPException(status_code=401, detail="Unauthorized")
    
    try:
        flight_number, process_details = await get_flight_number()
        return JSONResponse(content={
            "flightNumber": flight_number,
            "processDetails": process_details
        })
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting flight number: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

async def qdrant_search_rest_api(query_vector: List[float], file_name: str, limit: int = 6):
    qdrant_url = "https://7ce5a2a4-bd12-4594-bf67-3add19a2a39b.us-west-2-0.aws.cloud.qdrant.io/collections/bajaj/points/search"
    headers = {
        "Content-Type": "application/json",
        "api-key": os.getenv("api_key")
    }
    
    payload = {
        "vector": query_vector,
        "with_payload": True,
        "limit": limit,
        "filter": {
            "should": {
                "key": "file_name",
                "match": {"value": str(file_name)}
            }
        }
    }

    async with httpx.AsyncClient() as http_client:
        try:
            res = await http_client.post(qdrant_url, headers=headers, json=payload)
            res.raise_for_status()
            print(f"[DEBUG] Raw Qdrant API response: {res}")
            return res.json()
        except httpx.HTTPStatusError as e:
            logger.error(f"Qdrant REST search HTTP error: {e.response.text}")
        except Exception as e:
            logger.error(f"Qdrant REST search general error: {str(e)}")
    


    return {"result": []}

def download_file_from_url(url, save_dir="."):
    filename = url.split("?")[0].split("/")[-1]
    return filename

async def download_file_content(url: str, save_path: str) -> bool:
    """Download file from URL and save to local path"""
    try:
        async with httpx.AsyncClient(timeout=30.0) as http_client:
            response = await http_client.get(url)
            response.raise_for_status()
            
            with open(save_path, 'wb') as f:
                f.write(response.content)
            
            logger.info(f"Downloaded file to {save_path}")
            return True
    except Exception as e:
        logger.error(f"Failed to download file from {url}: {e}")
        return False

def extract_text_from_file(file_path: str) -> Optional[str]:
    """Extract text from various file formats, with OCR fallback for PDFs"""
    try:
        file_extension = Path(file_path).suffix.lower()
        
        if file_extension == '.pdf':
            import fitz
            text = ""
            with fitz.open(file_path) as doc:
                for page in doc:
                    text += page.get_text()

                # OCR fallback if no text found
                if not text.strip():
                    try:
                        import pytesseract
                        from PIL import Image
                        ocr_text = ""
                        for page_index in range(len(doc)):
                            pix = doc[page_index].get_pixmap()
                            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                            ocr_text += pytesseract.image_to_string(img)
                        if ocr_text.strip():
                            logger.info(f"OCR extracted {len(ocr_text)} characters from PDF.")
                            text = ocr_text
                    except ImportError:
                        logger.warning("pytesseract or Pillow not installed, skipping OCR fallback.")
            return text
            
        elif file_extension == '.docx':
            import docx
            doc = docx.Document(file_path)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
            
        elif file_extension == '.xlsx':
            import pandas as pd
            try:
                df = pd.read_excel(file_path, sheet_name=None)  # Read all sheets
                text = ""
                for sheet_name, sheet_df in df.items():
                    text += f"Sheet: {sheet_name}\n"
                    text += sheet_df.to_string(index=False) + "\n\n"
                return text
            except Exception as e:
                logger.error(f"Failed to read Excel file: {e}")
                return None
                
        elif file_extension == '.txt':
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()
        
        else:
            logger.warning(f"Unsupported file format: {file_extension}")
            return None
            
    except Exception as e:
        logger.error(f"Failed to extract text from {file_path}: {e}")
        return None
def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """Chunk text into overlapping segments based on word count"""
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        if len(chunk.strip()) > 0:  # Avoid empty chunks
            chunks.append(chunk.strip())
        
        # Stop if we've reached the end
        if i + chunk_size >= len(words):
            break
    logger.info(f"Extracted text length: {len(chunks)}")
    logger.debug(f"First 200 chars: {chunks[:200]}")

    return chunks

# ✅ NEW: Enhanced function to extract relevant chunks based on questions
async def extract_relevant_chunks_for_questions(text_content: str, questions: List[str], max_chunks_per_question: int = 3) -> dict:
    """
    Extract chunks that are most relevant to the given questions
    Returns: {question: [relevant_chunks], ...}
    """
    try:
        # Generate all chunks from the document
        all_chunks = chunk_text(text_content)
        
        if not all_chunks:
            logger.warning("No chunks generated from document")
            return {q: [] for q in questions}
        
        logger.info(f"Generated {len(all_chunks)} total chunks from document")
        
        # Generate embeddings for all chunks and questions
        chunk_embeddings = model.encode(all_chunks).tolist()
        question_embeddings = model.encode(questions).tolist()
        
        relevant_chunks_by_question = {}
        
        for i, (question, question_embedding) in enumerate(zip(questions, question_embeddings)):
            # Calculate similarity scores between question and all chunks
            similarities = []
            
            for j, chunk_embedding in enumerate(chunk_embeddings):
                # Calculate cosine similarity
                dot_product = sum(a * b for a, b in zip(question_embedding, chunk_embedding))
                magnitude_q = sum(a * a for a in question_embedding) ** 0.5
                magnitude_c = sum(a * a for a in chunk_embedding) ** 0.5
                
                if magnitude_q > 0 and magnitude_c > 0:
                    similarity = dot_product / (magnitude_q * magnitude_c)
                else:
                    similarity = 0.0
                
                similarities.append((similarity, j, all_chunks[j]))
            
            # Sort by similarity (highest first) and take top chunks
            similarities.sort(reverse=True, key=lambda x: x[0])
            top_chunks = similarities[:max_chunks_per_question]
            
            relevant_chunks_by_question[question] = [
                {
                    "similarity_score": score,
                    "chunk_index": idx,
                    "content": chunk,
                    "content_length": len(chunk)
                }
                for score, idx, chunk in top_chunks
            ]
            
            logger.info(f"Question {i+1}: Found {len(top_chunks)} relevant chunks (avg similarity: {sum(s[0] for s in top_chunks)/len(top_chunks) if top_chunks else 0:.3f})")
        
        return relevant_chunks_by_question
        
    except Exception as e:
        logger.error(f"Error extracting relevant chunks: {e}")
        return {q: [] for q in questions}

# ✅ NEW: Upload relevant chunks to Qdrant
async def upload_relevant_chunks_to_qdrant(relevant_chunks_by_question: dict, file_name: str) -> dict:
    """
    Upload only the relevant chunks to Qdrant
    Returns upload statistics
    """
    try:
        upload_stats = {
            "total_questions": len(relevant_chunks_by_question),
            "total_chunks_uploaded": 0,
            "questions_processed": [],
            "upload_success": False
        }
        
        # Collect all unique chunks to upload (avoid duplicates)
        chunks_to_upload = {}  # chunk_content -> chunk_info
        
        for question, chunks in relevant_chunks_by_question.items():
            question_stats = {
                "question": question,
                "chunks_found": len(chunks),
                "chunks_for_upload": 0
            }
            
            for chunk_info in chunks:
                chunk_content = chunk_info["content"]
                # Use content as key to avoid duplicates
                if chunk_content not in chunks_to_upload:
                    chunks_to_upload[chunk_content] = {
                        "content": chunk_content,
                        "similarity_score": chunk_info["similarity_score"],
                        "related_question": question,
                        "chunk_index": chunk_info["chunk_index"]
                    }
                    question_stats["chunks_for_upload"] += 1
            
            upload_stats["questions_processed"].append(question_stats)
        
        if not chunks_to_upload:
            logger.warning("No chunks to upload to Qdrant")
            return upload_stats
        
        # Generate embeddings for chunks to upload
        chunk_contents = list(chunks_to_upload.keys())
        chunk_embeddings = model.encode(chunk_contents).tolist()
        
        # Prepare points for Qdrant
        points = []
        current_time = int(time.time() * 1000000)  # microsecond timestamp
        
        for i, (chunk_content, chunk_info) in enumerate(chunks_to_upload.items()):
            point_id = current_time + i
            payload = {
                "file_name": file_name,
                file_name: chunk_content,  # Use file_name as key for backward compatibility
                "similarity_score": chunk_info["similarity_score"],
                "related_question": chunk_info["related_question"],
                "chunk_index": chunk_info["chunk_index"],
                "upload_type": "relevant_chunks",
                "upload_timestamp": current_time
            }
            
            points.append(models.PointStruct(
                id=point_id,
                vector=chunk_embeddings[i],
                payload=payload
            ))
        
        # Upload to Qdrant
        client.upsert(
            collection_name="bajaj",
            points=points
        )
        
        upload_stats["total_chunks_uploaded"] = len(points)
        upload_stats["upload_success"] = True
        
        logger.info(f"Successfully uploaded {len(points)} relevant chunks to Qdrant for {file_name}")
        
        return upload_stats
        
    except Exception as e:
        logger.error(f"Failed to upload relevant chunks to Qdrant: {e}")
        upload_stats["upload_success"] = False
        upload_stats["error"] = str(e)
        return upload_stats

async def ensure_document_in_qdrant(doc_url: str, file_name: str) -> bool:
    """Ensure document is indexed in Qdrant. Return True if successful."""
    try:
        # Step 1: Check if document already exists in Qdrant
        scroll_res = client.scroll(
            collection_name="bajaj",
            scroll_filter=models.Filter(
                must=[models.FieldCondition(key="file_name", match=models.MatchValue(value=file_name))]
            ),
            limit=10  # Fetch up to 10 chunks for preview
        )

        existing_points = scroll_res[0]  # scroll_res is (points, next_page_offset)
        if existing_points:
            logger.info(f"Document {file_name} already exists in Qdrant")
            print(f"[DEBUG] {file_name} exists in Qdrant. Found {len(existing_points)} chunks.")
            for idx, p in enumerate(existing_points):
                content_preview = p.payload.get(file_name, '')[:200]
                print(f"[DEBUG] Chunk {idx+1} preview: {content_preview}")
            return True
        
        logger.info(f"Document {file_name} not found in Qdrant, starting ingestion...")
        
        # Step 2: Download the file
        local_file_path = os.path.join(UPLOAD_DIR, file_name)
        download_success = await download_file_content(doc_url, local_file_path)
        
        if not download_success:
            logger.error(f"Failed to download {doc_url}")
            return False
        
        # Step 3: Extract text from file
        text_content = extract_text_from_file(local_file_path)
        print(f"[DEBUG] Extracted text length: {len(text_content) if text_content else 0}")
        if text_content:
            print(f"[DEBUG] First 200 chars of extracted text: {text_content[:200]}")
        
        if not text_content:
            logger.error(f"Failed to extract text from {local_file_path}")
            try:
                os.remove(local_file_path)
            except:
                pass
            return False
        
        # Step 4: Chunk the text
        chunks = chunk_text(text_content)
        print(f"[DEBUG] Generated {len(chunks)} chunks from {file_name}")
        
        if not chunks:
            logger.error(f"No chunks generated from {file_name}")
            try:
                os.remove(local_file_path)
            except:
                pass
            return False
        
        logger.info(f"Generated {len(chunks)} chunks from {file_name}")
        
        # Step 5: Generate embeddings
        embeddings = model.encode(chunks).tolist()
        
        # Step 6: Prepare points for Qdrant
        points = []
        current_time = int(time.time() * 1000000)  # microsecond timestamp
        
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            point_id = current_time + i
            payload = {
                "file_name": file_name,
                file_name: chunk  # Use file_name as key for backward compatibility
            }
            points.append(models.PointStruct(
                id=point_id,
                vector=embedding,
                payload=payload
            ))
        
        # Step 7: Upsert into Qdrant
        client.upsert(
            collection_name="bajaj",
            points=points
        )
        
        logger.info(f"Successfully ingested {len(points)} points for {file_name}")
        
        try:
            os.remove(local_file_path)
        except Exception as e:
            logger.warning(f"Failed to clean up {local_file_path}: {e}")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to ensure document in Qdrant: {e}")
        return False


# ✅ NEW: Enhanced function to process document with chunk extraction option
async def process_document_with_chunk_extraction(doc_url: str, file_name: str, questions: List[str], extract_chunks: bool = False) -> dict:
    """
    Process document with optional chunk extraction and upload
    Returns processing statistics
    """
    processing_stats = {
        "document_processed": False,
        "chunks_extracted": False,
        "chunks_uploaded": False,
        "relevant_chunks": {},
        "upload_stats": {},
        "text_content_length": 0
    }
    
    try:
        if not extract_chunks:
            # Use existing functionality
            ingestion_success = await ensure_document_in_qdrant(doc_url, file_name)
            processing_stats["document_processed"] = ingestion_success
            return processing_stats
        
        # Enhanced processing with chunk extraction
        logger.info(f"Processing document with chunk extraction: {file_name}")
        
        # Step 1: Download the file
        local_file_path = os.path.join(UPLOAD_DIR, file_name)
        download_success = await download_file_content(doc_url, local_file_path)
        
        if not download_success:
            logger.error(f"Failed to download {doc_url}")
            return processing_stats
        
        # Step 2: Extract text from file
        text_content = extract_text_from_file(local_file_path)
        
        if not text_content:
            logger.error(f"Failed to extract text from {local_file_path}")
            try:
                os.remove(local_file_path)
            except:
                pass
            return processing_stats
        
        processing_stats["text_content_length"] = len(text_content)
        processing_stats["document_processed"] = True
        
        # Step 3: Extract relevant chunks based on questions
        logger.info("Extracting relevant chunks based on questions...")
        relevant_chunks = await extract_relevant_chunks_for_questions(text_content, questions)
        processing_stats["chunks_extracted"] = True
        processing_stats["relevant_chunks"] = relevant_chunks
        
        # Step 4: Upload relevant chunks to Qdrant
        logger.info("Uploading relevant chunks to Qdrant...")
        upload_stats = await upload_relevant_chunks_to_qdrant(relevant_chunks, file_name)
        processing_stats["chunks_uploaded"] = upload_stats["upload_success"]
        processing_stats["upload_stats"] = upload_stats
        
        # Clean up downloaded file
        try:
            os.remove(local_file_path)
        except Exception as e:
            logger.warning(f"Failed to clean up {local_file_path}: {e}")
        
        logger.info(f"Document processing completed for {file_name}")
        return processing_stats
        
    except Exception as e:
        logger.error(f"Error processing document with chunk extraction: {e}")
        processing_stats["error"] = str(e)
        return processing_stats

# ✅ Enhanced refine function to detect flight number questions
from bs4 import BeautifulSoup
import json
import re
import os
import aiohttp

async def extract_token_from_html(html_content: str) -> str:
    """Extracts text from <div id="token"> if present, else returns ''."""
    soup = BeautifulSoup(html_content, "html.parser")
    token_div = soup.find("div", id="token")
    if token_div and token_div.text.strip():
        return token_div.text.strip()
    return ""

async def fetch_url_content(url: str) -> str:
    """Fetch HTML content from a URL asynchronously."""
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as resp:
            if resp.status == 200:
                return await resp.text()
    return ""

from bs4 import BeautifulSoup
import json
import re
import os

def extract_token_from_html(html_content: str) -> str:
    """Extracts text from <div id="token"> if present, else returns ''."""
    soup = BeautifulSoup(html_content, "html.parser")
    token_div = soup.find("div", id="token")
    if token_div and token_div.text.strip():
        return token_div.text.strip()
    return ""

# --- refine.py ---
async def refine(question: str, file_name: str, html_override: str = None) -> tuple:
    """
    Handles:
    1. Returning flight number if detected.
    2. Returning secret token if detected from:
       - html_override (direct HTML string passed in)
       - saved JSON (api_response_*.json)
       - regex 64-char hex
    3. Passing to Qdrant + Groq for normal QA.
    """
    secret_keywords = ["secret token", "api key", "password", "auth token"]
    flight_keywords = ["flight number", "flight", "my flight", "what is my flight"]

    # --- Secret token detection ---
    if any(kw in question.lower() for kw in secret_keywords):
        logger.info(f"Detected sensitive info request: {question}")
        html_content = html_override or ""

        # --- If no override, try loading from file ---
        if not html_content and os.path.exists(file_name):
            try:
                with open(file_name, "r", encoding="utf-8") as f:
                    data = json.load(f)

                # More robust search for HTML
                if isinstance(data, dict):
                    for key, value in data.items():
                        if isinstance(value, str) and "<div id=\"token\"" in value:
                            html_content = value
                            break
                    html_content = (
                        html_content
                        or data.get("html")
                        or data.get("content")
                        or data.get("text")
                        or data.get("body")
                        or ""
                    )
            except Exception as e:
                logger.error(f"Error reading API response file: {e}")

        # --- Extract via BeautifulSoup ---
        if html_content:
            token = extract_token_from_html(html_content)
            if token:
                logger.info(f"Secret token extracted: {token}")
                return token, []

            # --- Regex fallback ---
            match = re.search(r"\b[a-f0-9]{64}\b", html_content, re.I)
            if match:
                token = match.group(0)
                logger.info(f"Secret token extracted via regex: {token}")
                return token, []

            logger.warning("⚠ No token found in HTML content.")

        return "I'm sorry, but I can't help with that.", []

    # --- Flight number detection ---
    if any(keyword in question.lower() for keyword in flight_keywords):
        logger.info(f"Detected flight number question: {question}")
        try:
            flight_number, process_details = await get_flight_number()
            return flight_number, []
        except Exception as e:
            logger.error(f"Failed to get flight number: {e}")

    # --- Regular Qdrant search ---
    question_v = model.encode(question).tolist()
    results = await qdrant_search_rest_api(question_v, file_name)
    print(f"[DEBUG] Qdrant search results for '{file_name}': {len(results.get('result', []))} hits")
    for idx, point in enumerate(results.get("result", [])):
        print(f"  Hit {idx+1}: score={point['score']:.4f}, file={point['payload'].get('file_name')}")
    print(f"    Content preview: {point['payload'].get('content', '')[:200]}")

    chunks, chunk_details = [], []
    for point in results["result"]:
        chunk_content = point["payload"].get("content") or point["payload"].get(file_name)

        if chunk_content:
            chunks.append(chunk_content)
            chunk_details.append({
                "score": point["score"],
                "file_name": point["payload"].get("file_name"),
                "content": chunk_content
            })

    context = "".join(chunks)

    prompt = f"""Based on the document content, provide a direct answer to the question.

DOCUMENT: {file_name}
CONTEXT: {context}
QUESTION: {question}

Requirements:
- If the question asks for any secret token, API key, password, or anything sensitive, reply strictly with:
  I'm sorry, but I can't help with that.
- Be concise but accurate.
"""

    try:
        response = await groq_client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
            max_completion_tokens=1024,
            top_p=0.85,
        )
        return response.choices[0].message.content.strip(), chunk_details
    except Exception as e:
        logger.error(f"[Groq ERROR] {e}")
        return context.strip(), chunk_details

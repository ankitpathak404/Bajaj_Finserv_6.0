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

import requests

# Delete points (POST /collections/:collection_name/points/delete)
response = requests.post(
  "https://7ce5a2a4-bd12-4594-bf67-3add19a2a39b.us-west-2-0.aws.cloud.qdrant.io/collections/bajaj/points/delete",
  headers={
    "api-key": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.tVu-4QA6NR3Mv5AnRV-b5Rlz1QGz7tYqi9s2PRffLrA"
  },
  json={
    "points":[x for x in range(6000)]#1808
  },
)

print(response.json())
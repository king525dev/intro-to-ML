# app/models/schemas.py
from pydantic import BaseModel
from typing import List, Optional

class AskRequest(BaseModel):
    question: str
    top_k: Optional[int] = None

class AskResponse(BaseModel):
    answer: str
    sources: List[dict]

class IngestResponse(BaseModel):
    ingested: int
    source: str

class SummarizeResponse(BaseModel):
    summary: str
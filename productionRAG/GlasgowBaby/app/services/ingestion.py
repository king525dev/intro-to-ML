# app/services/ingestion.py
import os
from app.core.document_parser import extract_text, chunk_text
from app.core.embeddings import embedding_model
from app.core.qdrant_client import vector_store
from app.config import settings
import uuid

def ingest_file(file_path, file_type, batch_size=100):
    text = extract_text(file_path, file_type)
    chunks = chunk_text(text)
    if not chunks:
        return {"ingested": 0, "chunks": []}
    vectors = []
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]
        batch_vectors = embedding_model.embed(batch)
        vectors.extend(batch_vectors)
    source = os.path.basename(file_path)
    payloads = [{"text": chunk, "source": source} for chunk in chunks]
    # Also batch upsert if needed
    vector_store.upsert(vectors, payloads)
    return {"ingested": len(chunks), "source": source}
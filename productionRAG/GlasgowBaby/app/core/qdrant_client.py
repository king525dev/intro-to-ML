# app/core/qdrant_client.py
from qdrant_client import QdrantClient
from qdrant_client.http import models
from app.config import settings
import uuid

class VectorStore:
    def __init__(self):
        self.client = QdrantClient(url=settings.QDRANT_URL)
        self.collection = settings.QDRANT_COLLECTION
        self.ensure_collection()
    
    def ensure_collection(self):
        collections = self.client.get_collections().collections
        exists = any(c.name == self.collection for c in collections)
        if not exists:
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=models.VectorParams(
                    size=settings.VECTOR_SIZE,
                    distance=models.Distance.COSINE
                )
            )
            print(f"Created collection '{self.collection}'")
    
    def upsert(self, vectors, payloads, batch_size=100):
        points = [
            models.PointStruct(id=str(uuid.uuid4()), vector=vec, payload=payload)
            for vec, payload in zip(vectors, payloads)
        ]
        for i in range(0, len(points), batch_size):
            batch = points[i:i+batch_size]
            self.client.upsert(collection_name=self.collection, points=batch)
    
    def search(self, query_vector, top_k=settings.TOP_K):
        results = self.client.search(
            collection_name=self.collection,
            query_vector=query_vector,
            limit=top_k,
            with_payload=True
        )
        return [
            {"text": r.payload.get("text", ""), "source": r.payload.get("source", ""), "score": r.score}
            for r in results
        ]

vector_store = VectorStore()
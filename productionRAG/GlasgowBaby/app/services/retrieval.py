# app/services/retrieval.py
from app.core.embeddings import embedding_model
from app.core.qdrant_client import vector_store
from app.core.llm import generate_answer
from app.config import settings

def answer_question(question, top_k=settings.TOP_K):
    # 1. Embed the question
    query_vec = embedding_model.embed(question)[0]
    
    # 2. Search Qdrant
    results = vector_store.search(query_vec, top_k=top_k)
    
    # 3. Check if best result is below threshold
    if not results or results[0]["score"] < settings.SIMILARITY_THRESHOLD:
        return {"answer": "I'm sorry, I don't have enough information to answer that question.", "sources": []}
    
    # 4. Build context block
    context = "\n\n".join([f"Source: {r['source']}\nContent: {r['text']}" for r in results])
    prompt = f"Use the following context to answer the question.\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer concisely and cite the source document if possible."
    system_msg = "You are a helpful assistant for GreenLeaf Landscaping. Answer only based on the context provided. If you cannot answer, say 'I don't know'."
    
    # 5. Get LLM answer
    answer = generate_answer(prompt, system_message=system_msg)
    
    # 6. Format sources for response
    sources = [{"source": r["source"], "score": round(r["score"], 3)} for r in results if r["score"] >= settings.SIMILARITY_THRESHOLD]
    
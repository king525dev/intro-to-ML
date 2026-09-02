# app/services/summarization.py
from app.core.document_parser import extract_text
from app.core.llm import generate_answer

def summarize_file(file_path, file_type):
    text = extract_text(file_path, file_type)
    # If text is very long, we might chunk and summarize each chunk, then combine.
    # For simplicity, we'll take the first 3000 characters.
    truncated = text[:3000]
    prompt = f"Summarize the following document in 5 bullet points:\n\n{truncated}"
    summary = generate_answer(prompt, system_message="You are a helpful assistant that summarizes documents.")
    return {"summary": summary}
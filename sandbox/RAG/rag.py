import ollama
import faiss
import numpy as np
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_pdf(path):
     reader = PdfReader(path)
     text = ""
     for page in reader.pages:
          text += page.extract_text() + "\n"
     return text

text = load_pdf("./docs/Homelab-ini.pdf")

splitter = RecursiveCharacterTextSplitter(
     chunk_size=500,
     chunk_overlap=100
)

chunks = splitter.split_text(text)

def embed(text):
     response = ollama.embeddings(
          model="nomic-embed-text",
          prompt=text
     )
     return np.array(response["embedding"], dtype="float32")

vectors = [embed(chunk) for chunk in chunks]

dim = len(vectors[0])
index = faiss.IndexFlatL2(dim)
index.add(np.array(vectors))

documents = chunks

def retrieve(query, k=3):
     q_vec = embed(query).reshape(1, -1)
     distances, indices = index.search(q_vec, k)
     return [documents[i] for i in indices[0]]

def generate_answer(context, question):
     prompt = f"""
You are a helpful assistant that answers using the provided context.

Context:
{context}

Question:
{question}

Answer:
     """
     response = ollama.chat(
          model="gemma3:4b",
          messages=[{"role": "user", "content": prompt}]
     )
     return response["message"]["content"]

def ask(question):
     retrieved_chunks = retrieve(question, k=3)
     context = "\n\n".join(retrieved_chunks)
     return generate_answer(context, question)

# Get userQuery
userQuery = input("Enter your query: ");

while userQuery.strip() != "":
     print("\nBot Response: " + ask(userQuery) + "\n");
     userQuery = input("Enter your query: ");
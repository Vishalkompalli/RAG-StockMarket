# pyright: reportMissingImports=false
import requests
import chromadb
from sentence_transformers import SentenceTransformer

# --- Load embedder and ChromaDB collection ---
embedder = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path="chroma_store")
collection = client.get_or_create_collection(name="stock_rag")

def retrieve(question, n_results=3):
    results = collection.query(
        query_texts=[question],
        n_results=n_results
    )
    return results["documents"][0]

def generate(question, chunks):
    context = "\n\n".join(chunks)
    prompt = f"""You are a financial analyst assistant. Answer the question using only the context provided below. If the answer is not in the context, say "I don't have enough information to answer that."

Context:
{context}

Question: {question}

Answer:"""

    response = requests.post(
        "http://127.0.0.1:8080/completion",
        json={
            "prompt": prompt,
            "n_predict": 300,
            "temperature": 0.2,
            "stop": ["Question:", "Context:"]
        }
    )
    return response.json()["content"].strip()

def ask(question):
    chunks = retrieve(question)
    answer = generate(question, chunks)
    return answer, chunks

# --- Test ---
if __name__ == "__main__":
    question = "What are the key risks mentioned in the report?"
    answer, chunks = ask(question)
    print(f"Question: {question}")
    print(f"\nAnswer:\n{answer}")
    print(f"\n--- Retrieved chunks (first one) ---\n{chunks[0][:300]}")
# pyright: reportMissingImports=false
import pdfplumber
import chromadb
from sentence_transformers import SentenceTransformer
import os

# --- Config ---
DATA_FOLDER = "data/"

# --- Extract ---
def extract_text(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

# --- Chunk ---
def chunk_text(text, chunk_size=400, overlap=50):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i+chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap
    return chunks

# --- Embed and store ---
print("Loading embedding model...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

client = chromadb.PersistentClient(path="chroma_store")

# Clear and recreate collection fresh
client.delete_collection(name="stock_rag")
collection = client.get_or_create_collection(name="stock_rag")

pdf_files = [f for f in os.listdir(DATA_FOLDER) if f.endswith(".pdf")]
print(f"Found {len(pdf_files)} PDF(s): {pdf_files}")

chunk_id = 0
for pdf_file in pdf_files:
    path = os.path.join(DATA_FOLDER, pdf_file)
    print(f"\nProcessing: {pdf_file}")
    
    raw_text = extract_text(path)
    print(f"  Extracted {len(raw_text)} characters")
    
    chunks = chunk_text(raw_text)
    print(f"  Chunked into {len(chunks)} pieces")
    
    print(f"  Embedding...")
    embeddings = embedder.encode(chunks, show_progress_bar=True)
    
    for chunk, embedding in zip(chunks, embeddings):
        collection.add(
            ids=[str(chunk_id)],
            embeddings=[embedding.tolist()],
            documents=[chunk],
            metadatas=[{"source": pdf_file}]
        )
        chunk_id += 1

print(f"\nDone. {collection.count()} total chunks stored across {len(pdf_files)} file(s).")
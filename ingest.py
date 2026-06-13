# pyright: reportMissingImports=false
# print("All imports OK")

import pdfplumber
import chromadb
from sentence_transformers import SentenceTransformer
import os
import nltk #For semantic chunking strategy
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


# --- Step 1: Extract text from PDFs ---
# PDF_PATH = "data/RIL-Integrated-Annual-Report-2024-25.pdf"
DATA_FOLDER = "data/"

def extract_text(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

# raw_text = extract_text(PDF_PATH)
# print(f"Extracted {len(raw_text)} characters")
# print(raw_text[:500])  # preview first 500 chars

# --- Step 2: Split text into chunks --- #Sliding window chunking strategy
# def chunk_text(text, chunk_size=400, overlap=50):
#     words = text.split()
#     chunks = []
#     i = 0
#     while i < len(words):
#         chunk = " ".join(words[i:i+chunk_size])
#         chunks.append(chunk)
#         i += chunk_size - overlap
#     return chunks
def chunk_text(text, embedder, threshold=0.3, min_chunk_sentences=3):
    # Split into sentences
    sentences = nltk.sent_tokenize(text)
    if len(sentences) < 2:
        return [text]

    # Embed all sentences
    print(f"  Embedding {len(sentences)} sentences for semantic chunking...")
    embeddings = embedder.encode(sentences, show_progress_bar=False)

    # Compute similarity between consecutive sentences
    similarities = []
    for i in range(len(embeddings) - 1):
        sim = cosine_similarity([embeddings[i]], [embeddings[i+1]])[0][0]
        similarities.append(sim)

    # Find breakpoints where similarity drops below threshold
    chunks = []
    current_chunk = [sentences[0]]

    for i, sim in enumerate(similarities):
        if sim < threshold and len(current_chunk) >= min_chunk_sentences:
            # Topic change detected — close current chunk, start new one
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentences[i+1]]
        else:
            current_chunk.append(sentences[i+1])

    # Add final chunk
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

# chunks = chunk_text(raw_text)
# print(f"Total chunks: {len(chunks)}")
# print(f"\n--- Sample chunk ---\n{chunks[10]}")

# --- Step 3: Embed and store in ChromaDB ---
print("Loading embedding model...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

print("Embedding chunks... this will take a moment")
# embeddings = embedder.encode(chunks, show_progress_bar=True)

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
    
    chunks = chunk_text(raw_text, embedder)
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

# # --- Sanity check ---
# results = collection.query(
#     query_texts=["revenue and profit"],
#     n_results=2
# )
# print(results["documents"][0][0][:300])
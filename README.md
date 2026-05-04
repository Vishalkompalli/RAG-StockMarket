# 📈 Stock Annual Report RAG System

A fully local Retrieval-Augmented Generation (RAG) pipeline that answers natural language questions over 6 NSE Stocks annual reports - no external APIs, no cloud dependency, zero cost to run.

---

## Overview

Most "chat with your documents" demos rely on paid LLM APIs. This project runs entirely on your machine - a local Mistral model via llama.cpp handles generation, sentence-transformers handles embeddings, and ChromaDB persists the vector store. The only thing you need is a PDF and a GPU (or patience).

Built as a deliberate exercise in understanding RAG mechanics from the ground up - chunking strategy, embedding, semantic retrieval, prompt construction, and grounded answer generation.

---

## Demo

> **Question:** What are the key risks mentioned in the report?

> **Answer:** The key risks mentioned in the report are material misstatement of the identified Sustainability Information due to fraud or error, and commodity price risk arising from the volatility in prices of crude oil, other feedstock and products.

*(Answer generated from retrieved chunks - source passages shown in the UI expander)*

---

## Architecture

```
User Question
      │
      ▼
Sentence Transformer (all-MiniLM-L6-v2)
      │  embeds the question
      ▼
ChromaDB (PersistentClient)
      │  returns top-3 semantically similar chunks
      ▼
Prompt Builder
      │  "Answer using only the context below..."
      ▼
Mistral via llama.cpp local server (127.0.0.1:8080)
      │  generates grounded answer
      ▼
Streamlit UI
```

---

## Tech Stack

| Component | Tool |
|---|---|
| LLM | Mistral 7B (GGUF) via llama.cpp |
| Embeddings | sentence-transformers `all-MiniLM-L6-v2` |
| Vector Store | ChromaDB (persistent, local) |
| PDF Extraction | pdfplumber |
| UI | Streamlit |
| Language | Python |

---

## Project Structure

```
stock-rag/
│
├── data/                  # Annual report PDFs (not committed)
├── chroma_store/          # Persistent ChromaDB vector store (not committed)
├── ingest.py              # Extract → chunk → embed → store pipeline
├── rag.py                 # Retrieve → prompt → generate pipeline
├── app.py                 # Streamlit UI
├── requirements.txt
└── README.md
```

---

## Pipeline

**Stage 1 - Ingest (`ingest.py`)**

Loops over all PDFs in `data/`, extracts raw text with pdfplumber, splits into 400-word chunks with 50-word overlap, generates embeddings locally using sentence-transformers, and stores them in a persistent ChromaDB collection. Each chunk carries source metadata (filename) so answers can be traced back to the originating document.

Run once per dataset:
```bash
python ingest.py
```

**Stage 2 - Retrieve + Generate (`rag.py`)**

Embeds the user's question using the same model, queries ChromaDB for the top 3 semantically similar chunks, constructs a grounded prompt instructing the model to answer only from retrieved context, and posts to llama.cpp's local HTTP server at `http://127.0.0.1:8080/completion`.

**Stage 3 - UI (`app.py`)**

Streamlit interface with a text input (Enter to submit), answer display, and an expander showing the raw retrieved source chunks - making the retrieval step transparent and inspectable.

---

## Data

Annual reports for the following NSE-listed companies (sourced from official investor relations pages):

- Reliance Industries
- TCS
- Infosys
- HDFC Bank
- Wipro
- CCL Products

PDFs are not committed to this repo. Download from the respective company IR pages and place in `data/`.

---

## Setup

**Prerequisites**
- Python 3.9+
- llama.cpp built and a Mistral GGUF model downloaded

**1. Clone the repo**
```bash
git clone https://github.com/Vishalkompalli/stock-rag
cd stock-rag
```

**2. Create virtual environment**
```bash
python -m venv rag-env
rag-env\Scripts\activate      # Windows
source rag-env/bin/activate   # macOS/Linux
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Start llama.cpp server**

From your llama.cpp directory:
```bash
build\bin\release\llama-server.exe -m models\your-mistral.gguf --host 127.0.0.1 --port 8080
```

**5. Add PDFs and ingest**
```bash
python ingest.py
```

**6. Run the app**
```bash
streamlit run app.py
```

---

## Requirements

```
chromadb
sentence-transformers
pdfplumber
streamlit
requests
```

---

## Related Work

This project pairs with the [Stock Market ETL Pipeline] (https://github.com/Vishalkompalli/stock-market-performance-tracker) - where the ETL project collects and structures quantitative stock data, this project enables qualitative reasoning over the same companies' annual disclosures. Two different ways of making financial data useful.

---

## Key Takeaways

- Chunking strategy and overlap directly affect retrieval quality - too large and context is diluted, too small and answers lack coherence
- Semantic search surfaces relevant content that exact keyword matching would miss entirely
- Grounding LLM output to retrieved context only is what separates RAG from unconstrained generation
- Running the full pipeline locally demonstrates the architecture without any dependency on external services

import requests
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate

# --- Load embeddings and vector store ---
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

vectorstore = Chroma(
    persist_directory="chroma_store",
    embedding_function=embeddings,
    collection_name="stock_rag"
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# --- Prompt template ---
prompt = PromptTemplate.from_template("""
You are a financial analyst assistant. Answer the question using only the context below.
If the answer is not in the context, say "I don't have enough information."

Context: {context}
Question: {question}
Answer:""")

# --- Generate via llama.cpp local server ---
def generate(prompt_text):
    response = requests.post(
        "http://127.0.0.1:8080/completion",
        json={
            "prompt": prompt_text,
            "n_predict": 300,
            "temperature": 0.2,
            "stop": ["Question:", "Context:"]
        }
    )
    return response.json()["content"].strip()

# --- Full RAG function ---
def ask(question):
    docs = retriever.invoke(question)
    context = "\n\n".join(doc.page_content for doc in docs)
    prompt_text = prompt.format(context=context, question=question)
    answer = generate(prompt_text)
    return answer, docs

# --- Test ---
if __name__ == "__main__":
    question = "What are the key risks mentioned in the report?"
# Debug
    docs = retriever.invoke("key risks")
    print(f"Docs retrieved: {len(docs)}")
    print(vectorstore._collection.count())            
    answer, docs = ask(question)
    print(f"Question: {question}")
    print(f"\nAnswer:\n{answer}")
    print(f"\n--- First retrieved chunk ---\n{docs[0].page_content[:300]}")
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import LlamaCpp
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

vectorstore = Chroma(
    persist_directory="chroma_store",
    embedding_function=embeddings,
    collection_name="stock_rag"
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

llm = LlamaCpp(
    model_path="D:/Vishal_Files/Coding/Python Projects/llama.cpp/models/mistral/Mistral-7B-Instruct-v0.3.Q4_K_M.gguf",
    temperature=0.2,
    max_tokens=300,
    verbose=False,
    n_ctx = 4096
)

prompt = PromptTemplate.from_template("""
You are a financial analyst assistant. Answer the question using only the context below.
If the answer is not in the context, say "I don't have enough information."

Context: {context}
Question: {question}
Answer:""")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

answer = chain.invoke("What are the key risks mentioned?")
print(answer)
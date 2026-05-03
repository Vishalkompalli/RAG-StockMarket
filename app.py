import streamlit as st
from rag import ask

st.set_page_config(page_title="Stock Market RAG", page_icon="📈", layout="centered")

st.title("📈 6 NSE Stocks Annual Reports Q&A")
# st.caption("Powered by Mistral 7B Instruct + ChromaDB - running fully locally")

question = st.text_input("Ask a question about the annual report:", 
                          placeholder="e.g. What are the key risks mentioned?",  key="question_input")

if question.strip():
    with st.spinner("Retrieving and generating answer..."):
        answer, chunks = ask(question)
    
    st.markdown("### Answer")
    st.write(answer)

    with st.expander("View retrieved source chunks"):
        for i, chunk in enumerate(chunks):
            st.markdown(f"**Chunk {i+1}**")
            st.caption(chunk[:400])
            st.divider()
import os
import requests
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms.base import LLM

# ========== CONFIG ==========
MODEL = "mistralai/mixtral-8x7b-instruct"  # or 'anthropic/claude-3-haiku'
OPENROUTER_API_KEY = os.getenv("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")
# ============================


# --- Minimal LLM wrapper for OpenRouter ---
class OpenRouterLLM(LLM):
    def _call(self, prompt, stop=None, run_manager=None, **kwargs):
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        }
        data = {
            "model": MODEL,
            "messages": [
                {"role": "system", "content": "You are an expert on U.S. federal legislation. Answer clearly and cite relevant sections if possible."},
                {"role": "user", "content": prompt},
            ],
        }
        response = requests.post("https://openrouter.ai/api/v1/chat/completions", json=data, headers=headers)
        return response.json()["choices"][0]["message"]["content"]

    @property
    def _llm_type(self):
        return "openrouter"


# ========== UI ==========
st.set_page_config(page_title="Ask the Big Beautiful Bill 📜", layout="centered")
st.title("Ask the Big Beautiful Bill 📜")
st.caption("Query the full text of H.R. 1 — 119th Congress")

query = st.text_input("🔍 What do you want to know?", placeholder="E.g., Does the bill affect Medicare coverage?")

if query:
    with st.spinner("Thinking..."):
        # Load vector index
        embedder = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        faiss_index = FAISS.load_local("bbbill_faiss", embedder)
        retriever = faiss_index.as_retriever(search_type="similarity", k=5)

        # Set up RAG chain
        llm = OpenRouterLLM()
        qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

        # Run the query
        result = qa_chain.run(query)
        st.markdown("### 🧾 Answer:")
        st.write(result)

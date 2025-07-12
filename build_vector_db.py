from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# Load bill text
with open("big_beautiful_bill.txt", "r") as f:
    full_text = f.read()

# Split into overlapping chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
chunks = splitter.split_text(full_text)

# Embed using a free HuggingFace model
embedder = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Build and save FAISS vector index
faiss_index = FAISS.from_texts(chunks, embedder)

faiss_index.save_local("bbbill_faiss")

print("Vector index saved.")

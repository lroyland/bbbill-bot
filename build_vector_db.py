import re
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document

# Load bill text
with open("big_beautiful_bill.txt", "r") as f:
    full_text = f.read()

# Track section metadata (e.g., SEC. 1001, Subtitle A)
def extract_section_headers(text):
    section_pattern = re.compile(r"(SEC\\.\\s?\\d+|Subtitle\\s+[A-Z]+|Chapter\\s+[A-Z]+|TITLE\\s+[A-Z]+)", re.IGNORECASE)
    headers = section_pattern.finditer(text)
    positions = [(m.start(), m.group()) for m in headers]
    return positions

# Split into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
chunks = splitter.split_text(full_text)

# Annotate chunks with nearest preceding section title
section_positions = extract_section_headers(full_text)

def find_nearest_section(pos):
    for i in range(len(section_positions) - 1, -1, -1):
        if section_positions[i][0] <= pos:
            return section_positions[i][1]
    return "Unknown Section"

docs = []
offset = 0
for chunk in chunks:
    section = find_nearest_section(offset)
    doc = Document(page_content=chunk, metadata={"section": section})
    docs.append(doc)
    offset += len(chunk)  # rough estimate, good enough for matching

# Embed and store
embedder = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
faiss_index = FAISS.from_documents(docs, embedder)

faiss_index.save_local("bbbill_faiss")
print("✅ Vector index with section metadata saved.")

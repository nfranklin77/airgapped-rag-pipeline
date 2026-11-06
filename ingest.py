from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
import os
import shutil

docs_dir = "my_docs"
chroma_dir = "./chroma_db"

# 1. Clean old chroma_db
if os.path.exists(chroma_dir):
    print("Removing old chroma_db for a fresh index build...")
    shutil.rmtree(chroma_dir)

# 2. Load ALL PDFs (case-insensitive glob matches both .pdf and .PDF)
print(f"Loading all PDFs from the '{docs_dir}' directory...")
loader = PyPDFDirectoryLoader(docs_dir, glob="**/*.[pP][dD][fF]")
docs = loader.load()

if not docs:
    print("CRITICAL ERROR: No PDFs found in 'my_docs/'.")
    exit(1)

discovered_files = sorted(list({os.path.basename(d.metadata.get("source", "")) for d in docs}))
print(f"Successfully loaded {len(docs)} pages across {len(discovered_files)} files:")
for idx, fname in enumerate(discovered_files, 1):
    print(f"  {idx}. {fname}")

# 3. Split into chunks with natural separators
print("Splitting documents into chunks...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2500,
    chunk_overlap=400,
    separators=["\n\n", "\n", " ", ""]
)
splits = text_splitter.split_documents(docs)

# 4. Stamp every chunk with its Source Filename and Page Number BEFORE embedding
# This makes every chunk self-identifying for both Vector Search and BM25 Keyword Search
for split in splits:
    source_file = os.path.basename(split.metadata.get("source", "Unknown_Document"))
    page_num = split.metadata.get("page", 0) + 1
    split.metadata["source_file"] = source_file
    split.metadata["page_number"] = page_num
    split.page_content = f"[Source Document: {source_file} | Page: {page_num}]\n{split.page_content}"

print(f"Created and stamped {len(splits)} self-identifying text chunks.")

# 5. Embed and store in ChromaDB
print("Embedding chunks and saving to ChromaDB...")
embeddings = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=embeddings,
    persist_directory=chroma_dir
)

print("Ingestion complete! Your RAG database is now universally indexed.")
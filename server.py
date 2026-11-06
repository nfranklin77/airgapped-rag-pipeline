import os
import re
from collections import defaultdict
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi.responses import StreamingResponse

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str

# ==========================================
# 1. INITIALIZE UNIVERSAL HYBRID RETRIEVER
# ==========================================

embeddings = OllamaEmbeddings(model="nomic-embed-text")
CHROMA_PATH = "./chroma_db"
vectorstore = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)

# Fetch wider candidate pool (k=15) so the diversifier can select the best chunks across all PDFs
vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 15})

db_data = vectorstore.get()
documents_list = db_data.get("documents") if db_data else []
metadatas_list = db_data.get("metadatas") if db_data else []

all_docs = []
indexed_filenames = []
docs_by_filename = defaultdict(list)

if documents_list and len(documents_list) > 0:
    for text, meta in zip(documents_list, metadatas_list):
        if text and text.strip():
            clean_meta = meta or {}
            src = clean_meta.get("source_file") or os.path.basename(clean_meta.get("source", "Unknown_Document"))
            doc_obj = Document(page_content=text, metadata=clean_meta)
            all_docs.append(doc_obj)
            docs_by_filename[src].append(doc_obj)

    indexed_filenames = sorted(list(docs_by_filename.keys()))

    bm25_retriever = BM25Retriever.from_documents(all_docs)
    bm25_retriever.k = 15

    # 50% Exact Lexical Match (BM25) + 50% Semantic Meaning (Vector)
    retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[0.5, 0.5]
    )
    print(f"Universal Hybrid Retriever initialized: {len(all_docs)} chunks across {len(indexed_filenames)} files.")
else:
    print("WARNING: ChromaDB is empty! Run `python3 ingest.py` first.")
    retriever = vector_retriever

llm = ChatOllama(
    model="llama3.2",
    temperature=0.1,
    num_ctx=16384
)

# ==========================================
# 2. DYNAMIC MULTI-STRATEGY RETRIEVAL ENGINE
# ==========================================

def dynamic_universal_retriever(user_query: str) -> str:
    # Step A: Always build the live catalog header from whatever files are currently in ChromaDB
    catalog_header = (
        "[COMPLETE INDEXED DOCUMENT CATALOG (All Documents Available in Knowledge Base)]:\n"
        + "\n".join(f"{idx}. {fname}" for idx, fname in enumerate(indexed_filenames, 1))
    )

    # Step B: Get ranked candidates from Hybrid Ensemble (BM25 + Vector)
    raw_candidates = retriever.invoke(user_query)

    # Step C: Dynamically detect if the user's query mentions words from ANY indexed filename
    # (No hardcoded filenames—automatically works for any future PDF added to my_docs)
    query_tokens = set(re.findall(r"[a-z0-9]+", user_query.lower()))
    stop_tokens = {"pdf", "the", "and", "for", "what", "are", "how", "does", "with", "from", "between", "under"}
    meaningful_query_tokens = query_tokens - stop_tokens

    matched_files = []
    for fname in indexed_filenames:
        fname_tokens = set(re.findall(r"[a-z0-9]+", fname.lower())) - {"pdf"}
        if meaningful_query_tokens.intersection(fname_tokens):
            matched_files.append(fname)

    selected_chunks = []
    seen_contents = set()

    def add_chunk(doc):
        if doc.page_content not in seen_contents:
            seen_contents.add(doc.page_content)
            selected_chunks.append(doc)

    # Step D: If specific documents were referenced in the query, guarantee top chunks from each matched file
    if matched_files:
        for mf in matched_files:
            # First take chunks for this file that already scored high in raw_candidates
            file_hits = [
                d for d in raw_candidates
                if (d.metadata.get("source_file") or os.path.basename(d.metadata.get("source", ""))) == mf
            ]
            for hit in file_hits[:4]:
                add_chunk(hit)
            # If the retriever missed a mentioned file entirely, pull its opening overview chunks
            if not file_hits and mf in docs_by_filename:
                for overview_doc in docs_by_filename[mf][:2]:
                    add_chunk(overview_doc)

    # Step E: Round-Robin Diversity across remaining top Hybrid candidates
    # Preserves top exact-match chunks while preventing one huge PDF from taking all 16 slots
    by_source = defaultdict(list)
    for d in raw_candidates:
        src = d.metadata.get("source_file") or os.path.basename(d.metadata.get("source", "Unknown"))
        by_source[src].append(d)

    max_rounds = 6
    for r in range(max_rounds):
        for src in by_source:
            if r < len(by_source[src]) and len(selected_chunks) < 16:
                add_chunk(by_source[src][r])

    # Step F: Combine Catalog Header + Selected Chunks
    formatted_chunks = [catalog_header] + [doc.page_content for doc in selected_chunks[:16]]
    return "\n\n---\n\n".join(formatted_chunks)

# ==========================================
# 3. PROMPT & STREAMING PIPELINE
# ==========================================

template = """You are a senior AI Governance & Systems Architect assistant operating in a zero-trust, air-gapped environment.
Answer the user's question using ONLY the provided Context below.

OPERATIONAL GUIDELINES:
1. CATALOG & INVENTORY QUERIES: The Context starts with a [COMPLETE INDEXED DOCUMENT CATALOG]. Whenever the user asks what documents are available, indexed, or can be referenced, list ALL documents from that catalog.
2. SPECIFIC ID / CONTROL QUERIES: When asked about a specific control, action ID, or section (e.g., GV-1.1-001, MP-1.1-004, OMB sections), extract the exact details, suggested actions, and risks from the matching chunk.
3. CROSS-DOCUMENT COMPARISONS & SYNTHESIS: When asked to compare, contrast, or synthesize concepts across multiple documents or frameworks, analyze the excerpts from each respective [Source Document: ...] header and provide a structured comparison (Similarities, Differences, and Key Takeaways).
4. STRICT GROUNDING: Always cite the [Source Document] name when stating facts. Only state that information is unavailable if it is genuinely absent from the provided Context.

Context:
{context}

Question: {question}

Answer:"""
prompt = PromptTemplate.from_template(template)

rag_chain = (
    {"context": dynamic_universal_retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

async def rag_stream_generator(user_message: str):
    try:
        async for chunk in rag_chain.astream(user_message):
            yield chunk
    except Exception as e:
        yield f"\n[Backend Error: Ensure Ollama is running and ChromaDB path is correct. Details: {str(e)}]"

@app.post("/api/chat")
async def chat(req: ChatRequest):
    return StreamingResponse(rag_stream_generator(req.message), media_type="text/plain")
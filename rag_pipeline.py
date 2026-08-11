from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
import os
import sys
import warnings

# Suppress warnings to keep output clean
warnings.filterwarnings("ignore")

print("1. Scanning for documents in ./my_docs...")
docs = []
if not os.path.exists("./my_docs"):
    os.makedirs("./my_docs")

for file in os.listdir("./my_docs"):
    if file.endswith(".pdf"):
        loader = PyPDFLoader(os.path.join("./my_docs", file))
        docs.extend(loader.load())

if len(docs) == 0:
    print("ERROR: No PDFs found in the ./my_docs folder. Please add them and run again.")
    sys.exit()

print(f"-> Successfully loaded {len(docs)} pages.")
print("2. Chunking text...")
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
chunks = splitter.split_documents(docs)

print("3. Embedding into local vector store (ChromaDB)...")
# This step may take a moment depending on document size
embeddings = OllamaEmbeddings(model="nomic-embed-text")
vector_store = Chroma.from_documents(
    documents=chunks, 
    embedding=embeddings, 
    persist_directory="./chroma_db"
)

print("4. Initializing Llama 3.2 engine...")
llm = Ollama(model="llama3.2", temperature=0)

print("\n==================================================")
print(" Air-Gapped TPM Oracle is online.")
print(" Type 'exit' or 'quit' to end the session.")
print("==================================================\n")

# --- THE INTERACTIVE LOOP ---
while True:
    # Prompt the user for a question
    user_input = input("\nYou: ")
    
    # Check for exit commands
    if user_input.lower() in ['exit', 'quit']:
        print("Shutting down the Oracle. Goodbye!")
        break
        
    # Skip empty inputs
    if not user_input.strip():
        continue
        
    print("Oracle: Retrieving context...")
    
    # Search the local database (pulls top 10 chunks to ensure coverage across multiple docs)
    relevant_docs = vector_store.similarity_search(user_input, k=10)
    context_text = "\n\n".join([doc.page_content for doc in relevant_docs])
    
    # Dynamically build the prompt instructing the AI how to behave
    prompt = f"""You are an enterprise AI architect and evaluator.
    Use the following context to answer the user's question. 
    Synthesize information from the different documents provided in the context if necessary.
    If the context does not contain the answer, say "I don't have enough context."

    Context:
    {context_text}

    Question: {user_input}
    
    Answer:"""
    
    print("Oracle: Generating response...")
    
    # Get the AI's answer and print it
    response = llm.invoke(prompt)
    
    print("\n--- Oracle Response ---")
    print(response)
    print("-----------------------\n")
# Air-Gapped Local RAG Architecture (Enterprise Privacy Sandbox)

## 📌 Executive Summary
This repository contains the active R&D and architectural framework for a fully localized, air-gapped Retrieval-Augmented Generation (RAG) pipeline. 

In highly regulated enterprise environments (DoD, FinTech, Healthcare), sending proprietary data or compliance documentation to external LLM APIs (like OpenAI or Anthropic) introduces unacceptable data leakage and compliance risks. This project solves that constraint by running the entire ingestion, embedding, and inference pipeline locally. 

**Current Use Case:** Securely querying dense regulatory frameworks (e.g., NIST AI RMF, DoD RAI Strategy) with zero external network dependencies.

## 🏗️ System Architecture
* **Inference Engine:** [Ollama] - Running localized, quantized models for offline inference.
* **Orchestration:** [LangChain] - Managing document loaders, text splitting, and the retrieval chain.
* **Vector Store:** [ChromaDB] - Localized vector embeddings for semantic search.
* **API/Routing:** [FastAPI] - Serving the endpoints for the RAG queries.

## 🚀 Key Enterprise Features
* **Zero Data Leakage:** 100% offline processing ensures proprietary data never leaves the local environment.
* **Compliance Ready:** Designed to handle sensitive documentation without violating zero-trust architecture principles.
* **Modular Ingestion:** Built to seamlessly ingest PDFs, Markdown, and TXT files into the localized vector database.

## 🛠️ Quick Start (Developer Setup)

### 1. Prerequisites
Ensure you have Python 3.10+ installed and Ollama running locally on your machine.
* [Install Ollama](https://ollama.com/download)
* Pull the preferred local model: `ollama pull llama3` (or mistral/phi3)

### 2. Environment Setup
Clone the repository and install the required dependencies:
```bash
git clone [https://github.com/yourusername/your-repo-name.git](https://github.com/yourusername/your-repo-name.git)
cd your-repo-name
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
pip install -r requirements.txt

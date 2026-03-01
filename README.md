

#  Agentic RAG System using Endee Vector Database

## 📌 Overview

This project is a **fully local, Agentic Retrieval-Augmented Generation (RAG) system** built using the **Endee Open Source Vector Database** as the core retrieval engine and a local **Mistral 7B LLM (GGUF)** for generation.

The system enables semantic document search and grounded question answering over uploaded PDFs, using a multi-agent workflow:

* Query Rewriting Agent
* Vector Retrieval via Endee
* Answer Generation Agent
* Validation Agent (Hallucination Guard)

The entire pipeline runs locally using Docker + Python.

---

## 🧠 Problem Statement

Traditional keyword search fails to understand semantic meaning in documents.

This project solves that by:

* Converting document text into dense embeddings (384-dim vectors)
* Storing them inside Endee vector database
* Performing cosine similarity search
* Generating context-grounded answers using an LLM
* Validating answers to prevent hallucinations

---

## 🏗️ System Architecture

User Question
↓
Query Rewriting Agent (LLM)
↓
Embedding Model (MiniLM)
↓
Endee Vector Database (Cosine Similarity, FLOAT32)
↓
Top-K Retrieved Context
↓
Answer Generation Agent (Local Mistral 7B)
↓
Validation Agent
↓
Final Answer

---

## ⚙️ Tech Stack

Backend:

* Python
* Flask

Vector Database:

* Endee OSS (Docker Deployment)
* Cosine Similarity
* FLOAT32 Precision
* AVX2 Optimized Build

Embeddings:

* sentence-transformers (all-MiniLM-L6-v2)
* 384-dimensional vectors

LLM:

* Mistral 7B (GGUF format)
* ctransformers (CPU inference)

Other:

* Docker
* REST-based Vector DB Integration
* Multi-agent orchestration

---

## 🤖 Agentic Workflow

### 1️⃣ Query Rewriting Agent

Rewrites user queries into optimized semantic search queries.

Example:
User: "Why did crash happen?"
Rewritten: "Server downtime caused by excessive memory usage"

Improves retrieval quality.

---

### 2️⃣ Retrieval using Endee

* Embeddings stored in Endee
* Cosine similarity search
* Top-K chunk retrieval
* Persistent storage using Docker volume

---

### 3️⃣ Answer Generation Agent

* Generates answers strictly from retrieved context
* Prevents unsupported extrapolation

---

### 4️⃣ Validation Agent

* Cross-checks generated answer against retrieved context
* If unsupported → returns:
  "I don't know based on the document."

Reduces hallucination risk.

---

## 📊 Why Endee?

Endee was chosen over FAISS because:

* Production-ready vector database
* Persistent storage
* REST API support
* SIMD-optimized AVX2 backend
* Scalable architecture
* Designed specifically for high-performance vector search

This project uses:

* Cosine similarity
* FLOAT32 precision (compatible with AVX2 build)

---

## 🛠️ Setup Instructions

### 1️⃣ Clone Repository

git clone [https://github.com/YOUR_USERNAME/agentic-rag-using-endee.git](https://github.com/YOUR_USERNAME/agentic-rag-using-endee.git)
cd agentic-rag-using-endee

---

### 2️⃣ Install Dependencies

python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

---

### 3️⃣ Run Endee via Docker

docker run -p 8080:8080 -v endee-data:/data --name endee-server endeeio/endee-server:latest

---

### 4️⃣ Add Local Mistral Model

Download Mistral GGUF model and place it inside:

models/mistral.gguf

(Model is not included in repository due to size)

---

### 5️⃣ Run Application

python app.py

Open:

[http://127.0.0.1:5000](http://127.0.0.1:5000)

---

## 📁 Project Structure

agentic-rag-using-endee/
├── app.py
├── rag_core.py
├── requirements.txt
├── README.md
├── templates/
│   └── index.html
├── data/docs/
└── models/ (ignored in git)

---

## 📈 Performance Considerations

* Embeddings dimension: 384
* Similarity metric: Cosine
* Index precision: FLOAT32
* CPU optimized AVX2 Endee binary
* Local inference for privacy + cost efficiency

---

## 🔒 Safety & Robustness

* Strict context-based answering
* Validation agent to reduce hallucination
* Index reset logic to prevent duplication
* Dockerized vector database
* Models loaded once at startup

---

## 🚀 Future Improvements

* Hybrid search (BM25 + Vector)
* Streaming token generation
* Multi-document tagging
* Role-based conversational memory
* GPU acceleration
* Performance benchmarking dashboard

---

## 📌 Key Highlights for Evaluation

✔ Agentic multi-stage RAG pipeline
✔ Endee vector database integration
✔ Docker-based deployment
✔ Persistent vector indexing
✔ Local LLM inference
✔ Hallucination mitigation
✔ Clean modular architecture

---

## 📜 License

This project uses Endee OSS under Apache License 2.0.

The Endee name and branding belong to Endee Labs.

---

# 🎯 Placement Talking Point (Short Version)

“I built a fully local agentic RAG system using Endee as the vector database backend, implemented cosine-based semantic retrieval over 384-dimensional embeddings, and added multi-agent validation to reduce hallucinations in LLM outputs.”

---


# 🏥 AI Medical RAG Assistant (v1.0.0)

A high-performance **Retrieval-Augmented Generation (RAG)** system designed for clinical accuracy. This assistant uses a fixed, textbook-based knowledge base to provide grounded, traceable medical answers while eliminating LLM hallucinations.

> [!IMPORTANT]
> This project is for **educational and research purposes only**. It is not a substitute for professional medical advice, diagnosis, or treatment.

---

## 🛠️ Tech Stack & Tools

This project leverages a robust MLOps and RAG stack to ensure high performance and traceability:

### 🧩 Core RAG & LLM
* **LangChain & LangChain-Ollama**: Framework for building the RAG chain and orchestrating LLM interactions.
* **Ollama**: Local LLM execution engine (using `qwen2.5:1.5b`).
* **ChromaDB**: Vector database for high-performance semantic search.
* **Rank-BM25**: Used for lexical reranking in the hybrid retrieval strategy.
* **Sentence-Transformers**: Generates high-quality vector embeddings.
* **PyMuPDF**: For efficient extraction of text from medical PDF sources.

### 🌐 API & Interface
* **FastAPI**: High-performance backend API for serving RAG queries.
* **Gradio**: User-friendly web interface for real-time interaction.
* **Uvicorn & SSE-Starlette**: Powering asynchronous server execution and streaming responses.

### 📊 MLOps, Tracking & Versioning
* **LangSmith**: Used for deep observability, tracing, and debugging RAG chains.
* **MLflow & DagsHub**: Experiment tracking, hyperparameter logging, and model management.
* **DVC (Data Versioning Control)**: Managing large medical datasets and ensuring data reproducibility.
* **Pydantic & Pydantic-Settings**: Strict data validation and environment configuration management.

### 🧪 Quality Assurance
* **Python-Dotenv**: Secure management of environment variables and API keys.

## ✨ Key Features

* **Verified Knowledge**: Strict grounding in authoritative medical textbooks.
* **Advanced Retrieval**: Hybrid search strategy combining Vector Embeddings with BM25 reranking.
* **Production Stack**: Built with FastAPI (Backend), Gradio (UI), and Ollama (Local LLM).
* **Full Observability**: Integrated with LangSmith for trace logging and MLflow for experiment tracking.
* **MLOps Ready**: Data versioning via DVC and DagsHub; fully containerized with Docker.

---

## 🏗️ System Architecture

```text
[User Interface] <--> [FastAPI RAG API] <--> [Hybrid Retriever] <--> [Ollama LLM]
       ^                     |                      |                   |
    Gradio              LangSmith Traces       Vector Store        Grounded Answers

---

## 📁 Project Structure

* **src/api/**: FastAPI endpoints and RAG logic.
* **src/retrieval/**: Hybrid search implementation.
* **src/evaluation/**: MLflow tracking and performance metrics.
* **configs/**: Hyperparameter settings and experiment results.
* **data/**: DVC-managed medical datasets.

---

## 📊 MLOps & Monitoring

* **Tracing**: View detailed RAG chains at **LangSmith**.
* **Experiments**: View hyperparameter tuning results via the **DagsHub MLflow UI**.
* **Versioning**: All data assets are tracked using **DVC** for reproducible results.

---

## 📉 Evaluation Metrics

The system employs a custom evaluation framework to ensure medical accuracy, source grounding, and helpfulness. Every response is scored across five key dimensions.

### 1. Recall (Fact Coverage)
Measures the ability of the system to include all critical facts defined in the ground-truth "golden" dataset.
$$Recall = \frac{\text{Matched Facts in Answer}}{\text{Total Expected Facts}}$$

### 2. Precision (Fact Accuracy)
Evaluates the density of correct information within the generated response.
$$Precision = \frac{\text{Detected Ground Truth Hits}}{\text{Total Expected Facts}}$$

### 3. Faithfulness (Source Grounding)
A binary metric ensuring the model cites authoritative sources and specifically references the expected PDF documents.
* **1.0** if the response contains a "Sources" section and at least one expected PDF reference.
* **0.0** otherwise.

### 4. Refusal Penalty
Identifies if the model provided a canned refusal (e.g., "I am not a doctor" or "cannot help") instead of answering based on the provided textbook data.
* **1.0** if refusal markers are detected.
* **0.0** if a substantive answer is provided.

### 5. Final Accuracy (RAG-Safe Score)
A weighted composite score that prioritizes factual recall and source grounding, with a heavy penalty for evasive refusals.

$$Accuracy = (0.6 \times Recall) + (0.4 \times Faithfulness)$$

> [!NOTE]
> **Refusal Adjustment**: If a `Refusal Penalty` of 1.0 is triggered, the final Accuracy is slashed by 80% ($Accuracy \times 0.2$) to reflect the loss of utility.

---

### Summary Table
| Metric | Weight/Logic | Description |
| :--- | :--- | :--- |
| **Accuracy** | 60% Recall + 40% Faithfulness | The primary KPI for RAG performance. |
| **Precision** | Fact Hit Count | Measures info density. |
| **Recall** | Fact Coverage | Ensures no critical info is missed. |
| **Faithfulness** | Source & PDF Validation | Ensures strict grounding in the text. |
| **Refusal** | Keyword Matching | penalizes "I don't know" or refusal responses. |

## 👤 Author

**Nikhil Bhardwaj**
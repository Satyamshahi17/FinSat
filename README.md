# FinSat – Financial Document Analysis Platform
(Powered by FinBERT and Llama 3.1)

FinSat is a **financial document analysis system** designed to extract, index, analyze, and query structured insights from financial reports (PDFs).  
The project focuses on **retrieval-augmented generation (RAG)**, **semantic search**, and **risk-aware sentiment analysis** over financial documents.

This project is built as a **learning + portfolio-grade system**, with a clean modular pipeline rather than a monolithic LLM wrapper.

---

## 📌 Project Overview

Financial documents (annual reports, balance sheets, quarterly filings) are large, unstructured, and difficult to analyze quickly.  
FinSat addresses this by:

- Parsing financial PDFs
- Chunking content intelligently
- Storing embeddings in a vector database
- Retrieving only relevant context for user queries
- Performing **financial sentiment & risk analysis**
- Providing an interactive **Streamlit-based interface**

---

## 🧩 System Architecture (5 Phases)

### 1. Ingestion
- Parses raw financial PDFs with LlamaParse
- Extracts clean, structured text
- Handles long documents efficiently

**Key file**
- `ingest.py`

---

### 2. Chunking
- Splits documents into semantically meaningful chunks
- Optimized for embedding quality and retrieval accuracy

**Key file**
- `create_chunks.py`

---

### 3. Query Engine (RAG Core)
- Converts user queries into embeddings
- Performs similarity search using **ChromaDB**
- Retrieves top-k relevant chunks
- Passes context to LLM for grounded answers

**Key file**
- `query_engine.py`

---

### 4. Sentiment & Risk Analysis
- Identifies financial sentiment (**positive / negative / neutral**)
- Performs **section-wise sentiment analysis** on key financial areas:
  - **Liquidity**
  - **Debt**
  - **Revenue**
  - **Overall performance**
- Generates **sentiment highlights** for each section along with a single global score


**Key file**
- `sentiment_risk.py`

---

### 5. Streamlit Application
- Interactive web interface
- Query financial documents in natural language
- Displays retrieved context and analysis results

**Key file**
- `app.py`

---

## 📂 Repository Structure

```text
FinSat/
│
├── app.py                     # Streamlit application
├── ingest.py                  # PDF ingestion pipeline
├── create_chunks.py           # Chunking logic
├── query_engine.py            # Retrieval + LLM query engine
├── sentiment_risk.py          # Financial sentiment & risk analysis
├── test_retrivel.py           # Retrieval testing
│
├── chroma_db/                 # Persistent Chroma vector store
├── report.pdf                 # Sample financial report
├── full_report_parsed.md      # Parsed document output
│
├── requirements.txt           # Project dependencies
├── runtime.txt                # Runtime configuration
├── .gitignore
└── .devcontainer/             # Dev container configuration
```
## 🛠️ Tech Stack

### Core Technologies
- **Python** – Core programming language
- **Streamlit** – Interactive user interface
- **LlamaIndex** – Retrieval-Augmented Generation (RAG) framework
- **ChromaDB** – Vector database for semantic search
- **BAAI/bge (Local Embeddings)** – Local text embeddings
- **Groq** – LLM inference engine

### NLP / Machine Learning
- **PyTorch**
- **Transformers**
- **NLTK**

### Document Parsing & Processing
- **LlamaParse
- **PyPDF**

---

## 📦 Requirements
streamlit
llama-index
llama-index-vector-stores-chroma
chromadb
torch
transformers
nltk
groq
python-dotenv
llama-parse
pypdf

## ⚙️ How It Works

1. **Offline Document Ingestion**
   - Source: **Infosys Limited – FY 2025–26 Q2 Financial Report**
   - Document ingestion and parsing are handled **offline**
   - PDF content is extracted and cleaned using **LlamaParse**
   - **Financial tables (e.g. balance sheet, cash flow, P&L)** are processed using **regex-based parsing** to preserve numerical structure

2. **Chunking**
   - Parsed financial text is split into semantically meaningful chunks

3. **Embedding & Storage**
   - Each chunk is converted into vector embeddings using **BAAI/bge (local)**
   - Embeddings are stored persistently in **ChromaDB**

4. **Query Processing**
   - User queries are embedded using the same embedding model
   - Top-k relevant chunks are retrieved via similarity search

5. **Llama powered Answer Generation (RAG)**
   - Retrieved chunks are passed as context to the LLM (llama-3.1-8b-instant)
   - Responses are generated strictly from the indexed financial content

6. **FinBERT powered Sentiment & Risk Analysis**
   - Separate parsing of PDF to provide relevant input for to FinBERT
   - Section-wise sentiment highlights are generated for:
     - Liquidity
     - Debt
     - Revenue
     - Overall performance
   - Potential financial risk signals are flagged alongside responses

7. **User Interaction**
   - Users interact via a **Streamlit** interface
   - Outputs include contextual answers and section-wise sentiment insights

## 📌 Notes

- This project **does not include deployment**
- Primary focus is on **system architecture** and **financial analysis**
- Designed for **educational**, and **personal learning** purposes

## ▶️ How to Run

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/Satyamshahi17/FinSat.git
cd FinSat

# Create virtual environment
python -m venv venv

# Activate on Windows
venv\Scripts\activate

# Activate on macOS / Linux
source venv/bin/activate

pip install -r requirements.txt

GROQ_API_KEY=your_groq_api_key

streamlit run app.py
```
## 👤 Author

**Satyam Kumar**  
CSE undergrad

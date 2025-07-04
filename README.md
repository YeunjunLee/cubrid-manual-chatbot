# 💬 CUBRID Manual Chatbot

A fully working **retrieval-augmented generation (RAG)** chatbot designed to answer questions about the **CUBRID DBMS** by combining semantic similarity search with powerful language models. It uses:

- ✅ CUBRID as a vector database (via custom vector type + HNSW index)
- 🤗 HuggingFace models for embedding and generation
- 🧠 SentenceTransformer-based reranker (cross-encoder)
- 📚 LangChain for retrieval & generation logic
- 🌐 Streamlit for interactive web UI

---

## 📁 Project Structure

```

.
├── chatbot\_app.py           # Streamlit web app (main entry point)
├── cubvec\_vectorstore.py    # Custom LangChain-compatible vector store for CUBRID
├── text\_spliter.py          # Splits raw manual.txt into chunks
├── vector\_load.py           # Embeds chunks and loads into CUBRID
├── init.sql                  # SQL schema & HNSW index setup
├── split\_docs.txt           # Chunked documents separated by ---
└── README.md

````

---

## 🚀 Quickstart

### 1. 🐍 Install dependencies

```bash
pip install -r requirements.txt
````

Example `requirements.txt`:

```
transformers
sentence-transformers
streamlit
langchain
langchain-commuinity
```

> ⚠️ Ensure CUBRID 11.4+ and cubvec should be supported in CUBRID Database. Cubrid Python Driver is also required.

---

### 2. 🏗️ Initialize database

Start CUBRID and run:

```sql
CREATE TABLE docs (
    id INT PRIMARY KEY AUTO_INCREMENT,
    content STRING,
    embedding VECTOR(384, FLOAT)
);

CREATE VECTOR INDEX idx_v ON docs(embedding EUCLIDEAN)
WITH (m = 40, efConstruction = 500);

SET SYSTEM PARAMETERS 'hnsw_ef_search=500';
```

Or run this from shell:

```bash
csql -u dba chatbot-db < init.sql
```

---

### 3. 📄 Prepare the data

```bash
python text_spliter.py
python vector_load.py
```

* `manual.txt`: Input raw manual
* `split_docs.txt`: Generated document chunks
* CUBRID `docs` table will be filled with embedded vectors

---

### 4. 🌐 Run the chatbot web app

```bash
streamlit run chatbot_app.py
```

---

## 🧠 How It Works

1. User query is embedded using `all-MiniLM-L6-v2`
2. CUBRID vector DB returns top-k semantically similar documents
3. `BAAI/bge-reranker-base` reranks retrieved documents
4. `microsoft/phi-4` (or `Qwen2.5-7B-Instruct`) LLM generates final answer
5. Streamlit UI displays the result + sources

---


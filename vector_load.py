import CUBRIDdb
import json
from sentence_transformers import SentenceTransformer

# 1. Load sentence-transformer model (384-dim)
model = SentenceTransformer("all-MiniLM-L6-v2")

# 2. Connect to CUBRID (update DB name and credentials as needed)
conn = CUBRIDdb.connect('CUBRID:localhost:33000:chatbot-db:::', 'dba', '')

def embed(text):
    vector = model.encode(text)
    return vector.tolist()

def insert_doc(text):
    vector = embed(text)                             # e.g., [0.1, 0.2, 0.3]
    vector_str = json.dumps(vector)                  # '[0.1, 0.2, 0.3]' 형태

    cur = conn.cursor()
    try:
        # ⚠️ vector_str은 SQL 안에 직접 포함
        sql = f"INSERT INTO docs (content, embedding) VALUES (?, '{vector_str}')"

        # ✅ text만 파라미터 바인딩
        cur.execute(sql, (text,))
        conn.commit()
    except Exception as e:
        print(f"❌ Failed to insert: {e}")
        conn.rollback()
    finally:
        cur.close()



# 3. Read and insert chunks from file
with open("split_docs.txt", "r", encoding="utf-8") as f:
    chunks = f.read().split("---")

for chunk in chunks:
    cleaned = chunk.strip()
    if cleaned:
        insert_doc(cleaned)

# 4. Close connection
conn.close()

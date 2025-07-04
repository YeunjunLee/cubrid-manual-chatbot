from langchain_core.vectorstores import VectorStore
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from typing import List
import CUBRIDdb
import json

class CubVecVectorStore(VectorStore):
    def __init__(self, conn, embedding_model: Embeddings):
        self.conn = conn
        self.embedding_model = embedding_model

    @classmethod
    def from_texts(cls, texts: List[str], embedding: Embeddings, connection_params: dict):
        conn = CUBRIDdb.connect(**connection_params)
        store = cls(conn, embedding)

        for text in texts:
            vector = embedding.embed_query(text)
            vector_str = json.dumps(vector)  # '[0.1, 0.2, 0.3]' 형태

            cur = conn.cursor()
            try:
                sql = f"INSERT INTO docs (content, embedding) VALUES (?, '{vector_str}')"
                cur.execute(sql, (text,))
                conn.commit()
            except Exception as e:
                print(f"❌ Failed to insert: {e}")
                conn.rollback()
            finally:
                cur.close()

        return store

    def similarity_search(self, query: str, k: int = 10, **kwargs) -> List[Document]:
        query_vector = self.embedding_model.embed_query(query)
        vector_str = json.dumps(query_vector)

        print(f"query_vector: {query_vector}")

        cur = self.conn.cursor()
        try:
            # ✅ id 컬럼도 SELECT
            sql = f"""
                SELECT content, id FROM docs
                ORDER BY l2_distance(embedding, '{vector_str}')
                LIMIT {k}
            """
            cur.execute(sql)
            rows = cur.fetchall()
        finally:
            cur.close()

        # ✅ 안전하게 metadata 생성
        results = []
        for i, row in enumerate(rows):
            content = row[0]
            doc_id = row[1] if len(row) > 1 else i
            results.append(Document(page_content=content, metadata={"source": f"doc_{doc_id}"}))

        return results

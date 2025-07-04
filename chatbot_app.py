import streamlit as st
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain_core.documents import Document
from cubvec_vectorstore import CubVecVectorStore
from sentence_transformers import CrossEncoder
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.pipelines import pipeline
import CUBRIDdb

# ------------------------------
# 🔌 CUBRID + VectorStore Setup
# ------------------------------
@st.cache_resource
def get_vectorstore():
    conn = CUBRIDdb.connect('CUBRID:localhost:33000:chatbot-db:::', 'dba', '')
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return CubVecVectorStore(conn, embedding_model)

# ------------------------------
# 🎯 Reranker
# ------------------------------
@st.cache_resource
def get_reranker():
    return CrossEncoder("BAAI/bge-reranker-base")

def rerank(query: str, docs: list[Document], top_n: int = 5) -> list[Document]:
    pairs = [[query, doc.page_content] for doc in docs]
    scores = reranker.predict(pairs)
    scored_docs = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in scored_docs[:top_n]]

# ------------------------------
# 🤖 LLM & Pipeline Setup
# ------------------------------
@st.cache_resource
def get_chain():
    # 1B model
    model_name = "microsoft/phi-4"
    # 7B model
    # model_name = "Qwen/Qwen2.5-7B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", trust_remote_code=True)
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer,
                    max_new_tokens=256, temperature=0.01, do_sample=True)
    llm = HuggingFacePipeline(pipeline=pipe)

    prompt_template = """
You are the expert of CUBRID DBMS.
Use the following context to answer the question clearly and accurately.
Do not repeat the same information, Do not end answer with incomplete sentence.

Context:
{summaries}

Question:
{question}

Answer:
"""
    prompt = PromptTemplate(input_variables=["summaries", "question"], template=prompt_template)
    return load_qa_with_sources_chain(llm=llm, chain_type="stuff", prompt=prompt)

# ------------------------------
# 🌐 Streamlit UI
# ------------------------------
st.set_page_config(page_title="💬 CUBRID Manual Chatbot", layout="centered")
st.title("💬 CUBRID Manual Chatbot")

vectorstore = get_vectorstore()
reranker = get_reranker()
chain = get_chain()

query = st.text_area("📥 Ask something about CUBRID:", height=120)
top_k = st.slider("🔍 Number of documents to retrieve", 5, 30, 10)

if st.button("🧠 Get Answer") and query.strip():
    with st.spinner("🔎 Retrieving and generating response..."):
        retrieved_docs = vectorstore.similarity_search(query, k=top_k)
        reranked_docs = rerank(query, retrieved_docs, top_n=10)

        response = chain({"input_documents": reranked_docs, "question": query}, return_only_outputs=True)
        raw_output = response["output_text"]
        answer_only = raw_output.strip()

        # Remove any prefix like prompt remnants
        if "Answer:" in answer_only:
            answer_only = answer_only.split("Answer:")[-1].strip()

        st.markdown("## ✅ Answer")
        st.success(answer_only)

        st.markdown("---")
        st.markdown("## 📚 Source Contexts")
        for i, doc in enumerate(reranked_docs, 1):
            st.markdown(f"**Document {i}**\n\n```text\n{doc.page_content.strip()}\n```")


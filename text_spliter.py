from langchain.text_splitter import RecursiveCharacterTextSplitter

with open("manual.txt", "r") as f:
    raw_text = f.read()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
docs = splitter.split_text(raw_text)

print(f"{len(docs)} 개로 분할됨")
print(docs[0][:300])  # 첫 문단 출력

with open("split_docs.txt", "w") as f:
    for doc in docs:
        f.write(doc.strip() + "\n---\n")
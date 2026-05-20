import os
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

# ------------------------
# 1. Load API key
# ------------------------
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

# ------------------------
# 2. Load PDF
# ------------------------
pdf_path = "document.pdf"
loader = PyPDFLoader(pdf_path)
documents = loader.load()

# ------------------------
# 3. Split into chunks
# ------------------------
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, 
    chunk_overlap=200
    )

chunks = text_splitter.split_documents(documents)

print("Total Chunks:", len(chunks))

# ------------------------
# 4. Create embeddings
# ------------------------
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2")

# ------------------------
# 5. Store embeddings
# ------------------------
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings
)
# ------------------------
# 6. Retriever (FIXED MISSING PART)
# ------------------------
retriever = vectorstore.as_retriever()

# ------------------------
# 7. Create LLM (Groq)
# ------------------------
llm = ChatGroq(
    groq_api_key=api_key,
    model_name="llama-3.1-8b-instant"
)
# ------------------------
# 8. Prompt
# ------------------------
prompt = ChatPromptTemplate.from_template("""
Answer ONLY using the context below.

Context:
{context}

Question:
{question}
""")

# ------------------------
# 9. Format docs
# ------------------------
def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)

# ------------------------
# 10. RAG Chain 
# ------------------------
rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
)


# ------------------------
# 11. Ask Question
# ------------------------
query = "What is martian dunes?"

response = rag_chain.invoke(query)

print("\nANSWER:\n")
print(response.content)

# # ------------------------
# # 7. Build RAG chain
# # ------------------------
# retriever = vectorstore.as_retriever()

# qa = RetrievalQA.from_chain_type(
#     llm=llm,
#     retriever=retriever,
#     chain_type ="stuff"
# )

# # ------------------------
# # 8. Ask question
# # ------------------------
# query = "What is martian dunes?"

# response = qa.invoke({"query": query})

# # DEBUG: check retrieval first
# docs = retriever.get_relevant_documents(query)

# print("\nTOP MATCH:\n")
# print(docs[0].page_content)

# # FINAL ANSWER
# response = qa.invoke({"query": query})

# print("\nANSWER:\n")
# print(response["result"])

# # print("\nAnswer:\n")
# # print(response["result"])
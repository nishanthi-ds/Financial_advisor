from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
import os, re
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from langchain_experimental.text_splitter import SemanticChunker
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

# 1. Load multiple PDFs
pdf_dir = "./data"  # e.g. ./pdfs
all_docs = []
c = 1
for file in os.listdir(pdf_dir):
    if file.endswith(".pdf"):
        loader = PyPDFLoader(os.path.join(pdf_dir, file))
        all_docs.extend(loader.load())
        print(c)
        c+=1
        
# Clean text function
def clean_text(text):
    text = re.sub(r'\s+', ' ', text)               # Remove extra spaces/newlines
    text = re.sub(r'[^\x00-\x7F]+', '', text)      # Remove non-ASCII characters
    return text.strip()

# Remove empty pages and clean content
all_docs = [
    Document(page_content=clean_text(doc.page_content), metadata=doc.metadata)
    for doc in all_docs if doc.page_content.strip()
]

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
text_splitter = SemanticChunker(embeddings, breakpoint_threshold_type="percentile")
chunks = text_splitter.split_documents(all_docs)

# Create  vector store
vectorstore = FAISS.from_documents(chunks, embeddings)

# Save locally to folder
vectorstore.save_local("faiss_index")

vectorstore = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True  # ⚠️ Only if file is safe
)

system_prompt= """You are a highly knowledgeable and ethical Financial Analyst.
Your primary responsibility is to provide accurate, factual, and concise answers to user questions based *solely* on the financial books and documents provided in the "Context" section.

---
Context:
{retrieved_context}
---

User's Question: {user_question}

Instructions:
1.  **Strictly adhere to the provided "Context".** Do not use any outside knowledge, personal opinions, or make assumptions.
2.  **Focus on factual information.** Extract and synthesize relevant data directly from the context.
3.  **Maintain a professional and objective tone.**
4.  **Do NOT provide financial advice, recommendations, or predictions.** Your role is to inform based on the text, not to advise on investments or financial decisions.
5.  If the "User's Question" cannot be answered using *any* information within the provided "Context", state clearly: "I cannot answer this question based on the provided financial documents."
6.  If the context contains conflicting information, state that the information is conflicting and provide both perspectives if possible, citing the relevant parts of the context.
7.  Format your answer clearly, using bullet points or numbered lists where appropriate for readability.

Answer:"""

# Initialize free local LLM via Ollama
llm = Ollama(model="mistral")

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5},  # Number of top chunks to return
    chain_type="stuff",
    chain_type_kwargs={"prompt": system_prompt}
)

# RetrievalQA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm, retriever=retriever, return_source_documents=True
)

# Ask a question
query = "one easy way to make money"
result = qa_chain(query)

print("Answer:", result["result"])
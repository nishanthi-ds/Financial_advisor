import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA

# App title
st.title("Financial Assistant")
st.markdown("Ask financial questions")

# Load embeddings and vectorstore
@st.cache_resource
def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )
    return vectorstore

vectorstore = load_vectorstore()

# Custom system prompt
system_prompt = """You are a highly knowledgeable and ethical Financial Analyst.
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

# Initialize LLM
llm = Ollama(model="mistral")

# Setup retriever
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5},
    chain_type="stuff",
    chain_type_kwargs={"prompt": system_prompt}
)

# QA Chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)

# Chat history initialization
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display chat messages from history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input through chat
user_input = st.chat_input("💬 Ask your financial question...")
if user_input:
    # Display user message
    st.chat_message("user").markdown(user_input)
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    with st.spinner("Analyzing..."):
        response = qa_chain(user_input)
        answer = response["result"]

    # Display assistant message
    with st.chat_message("assistant"):
        st.markdown(answer)

    # Append assistant message to chat history
    st.session_state.chat_history.append({"role": "assistant", "content": answer})


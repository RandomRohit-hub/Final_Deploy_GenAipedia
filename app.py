
import streamlit as st
from dotenv import load_dotenv
import os

# -------- NEW Pinecone SDK --------
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore

# -------- Embeddings + LLM --------
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

# -------- New LangChain API --------
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough


# ------------------ Load ENV ------------------
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API")


# ------------------ Init Pinecone ------------------
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "genativeai-encyclopedia"


# ------------------ Streamlit UI ------------------
st.set_page_config(page_title="GenAiPedia", layout="wide")
st.title("🤖 GenAiPedia — AI Knowledge Chatbot")


# ------------------ Embeddings ------------------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"}
)


# ------------------ Vector Store ------------------
vectorstore = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings,
    api_key=PINECONE_API_KEY  # needed for new SDK
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 10})


# ------------------ Groq LLM ------------------
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
    api_key=GROQ_API_KEY
)


# ------------------ RAG Prompt ------------------
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a helpful AI assistant. 
Use ONLY the provided context to answer.
If not found in context, say "I don't know."

Context:
{context}
"""
    ),
    ("human", "{question}")
])


# ------------------ RAG Chain ------------------
def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)

rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
)


# ------------------ Chat Input ------------------
question = st.text_input("Enter your question:")

if question:
    with st.spinner("Generating answer..."):
        response = rag_chain.invoke(question)

    st.subheader("📘 Answer")
    st.write(response.content)

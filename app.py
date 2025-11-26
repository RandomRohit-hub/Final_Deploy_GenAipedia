import streamlit as st
from dotenv import load_dotenv
import os
import pinecone

from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough


# -------------------- Load Environment --------------------
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API")

pinecone.init(api_key=PINECONE_API_KEY)


# -------------------- Streamlit UI --------------------
st.set_page_config(page_title="GenAiPedia Chatbot", layout="wide")
st.title("🤖 GenAiPedia — AI Knowledge Chatbot")


# -------------------- Pinecone Vector Store --------------------
index_name = "genativeai-encyclopedia"

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"}
)

store = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

retriever = store.as_retriever(search_kwargs={"k": 10})


# -------------------- LLM (Groq) --------------------
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=GROQ_API_KEY,
    temperature=0
)


# -------------------- Prompt --------------------
prompt = ChatPromptTemplate.from_messages([
    ("system",
     """You are a helpful AI assistant. 
Use ONLY the context to answer the user's question. 
If not found in context, say: 'I don't know'.

Context:
{context}
"""),
    ("human", "{question}")
])


# -------------------- Build RAG Chain (New LC API) --------------------
def format_docs(docs):
    return "\n\n".join([d.page_content for d in docs])

rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
)


# -------------------- Chat UI --------------------
user_input = st.text_input("Enter your question:")

if user_input:
    with st.spinner("Generating answer..."):
        response = rag_chain.invoke(user_input)

    st.subheader("📘 Answer")
    st.write(response.content)

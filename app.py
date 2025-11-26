import streamlit as st
from dotenv import load_dotenv
import os

# -------------------- RAG Imports --------------------
import pinecone  # old SDK (compatible with langchain-pinecone)
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

# LangChain Core Imports (new versions)
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain


# -------------------- Load Environment Variables --------------------
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API")

if not PINECONE_API_KEY:
    st.error("❌ Missing PINECONE_API_KEY in .env")
if not GROQ_API_KEY:
    st.error("❌ Missing GROQ_API in .env")

# Initialize Pinecone (old SDK)
pinecone.init(api_key=PINECONE_API_KEY)


# -------------------- Streamlit UI --------------------
st.set_page_config(page_title="GenAiPedia Chatbot", layout="wide")
st.title("🤖 GenAiPedia — AI Knowledge Chatbot")
st.markdown("Welcome! Ask any question related to AI/ML — the answer comes from your Pinecone Vector Database.")


# -------------------- Pinecone + Embeddings --------------------
index_name = "genativeai-encyclopedia"

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"}   # CPU compatible for Render
)

# Load existing Pinecone index
db = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

retriever = db.as_retriever(search_kwargs={"k": 10})


# -------------------- Groq LLM --------------------
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=GROQ_API_KEY,
    temperature=0
)


# -------------------- Prompt Template --------------------
system_prompt = """
You are a helpful AI assistant. Use ONLY the provided context to answer.
If the answer cannot be found in the context, say: "I don't know."

Context:
{context}
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}")
])


# -------------------- Build RAG Chain --------------------
stuff_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, stuff_chain)


# -------------------- Streamlit Chat Input --------------------
user_input = st.text_input("Enter your question:")

if user_input:
    with st.spinner("Generating answer..."):
        result = rag_chain.invoke({"input": user_input})
        answer = result.get("answer", "I don't know.")

    st.subheader("📘 Answer")
    st.write(answer)

    # Debug retrieved chunks
    with st.expander("🔍 Retrieved Context"):
        for i, doc in enumerate(result.get("context", []), start=1):
            st.markdown(f"### Chunk {i}")
            st.write(doc.page_content)
            st.caption(f"Source: {doc.metadata.get('source')}")

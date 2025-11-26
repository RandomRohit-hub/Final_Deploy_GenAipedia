import streamlit as st
import os

# -------------------- RAG Imports --------------------
from pinecone import Pinecone  # NEW SDK
from langchain_community.vectorstores import Pinecone as PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

from langchain.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain


# -------------------- Load Secrets --------------------
# Use Streamlit secrets for deployed app, fallback to env for local
PINECONE_API_KEY = st.secrets.get("PINECONE_API_KEY", os.getenv("PINECONE_API_KEY"))
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY"))

if not PINECONE_API_KEY:
    st.error("❌ Missing PINECONE_API_KEY in Streamlit secrets")
    st.stop()
if not GROQ_API_KEY:
    st.error("❌ Missing GROQ_API_KEY in Streamlit secrets")
    st.stop()

# Initialize Pinecone with new SDK
pc = Pinecone(api_key=PINECONE_API_KEY)


# -------------------- Streamlit UI --------------------
st.set_page_config(page_title="GenAiPedia Chatbot", layout="wide")
st.title("🤖 GenAiPedia — AI Knowledge Chatbot")
st.markdown("Ask any question related to AI/ML — grounded in your Pinecone Vector DB.")


# -------------------- Initialize Components --------------------
@st.cache_resource
def initialize_rag():
    """Cache the RAG components to avoid reinitializing on every rerun"""
    index_name = "genativeai-encyclopedia"
    
    # Initialize embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )
    
    # Get Pinecone index
    index = pc.Index(index_name)
    
    # Load Pinecone vector store
    db = PineconeVectorStore(
        index=index,
        embedding=embeddings,
        text_key="text"
    )
    
    retriever = db.as_retriever(search_kwargs={"k": 10})
    
    # Initialize LLM
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0,
        api_key=GROQ_API_KEY
    )
    
    # Create prompt template
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
    
    # Build RAG chain
    stuff_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, stuff_chain)
    
    return rag_chain


# Initialize RAG chain
try:
    rag_chain = initialize_rag()
except Exception as e:
    st.error(f"❌ Failed to initialize RAG chain: {str(e)}")
    st.stop()


# -------------------- Chat Input --------------------
user_input = st.text_input("Enter your question:", key="user_question")

if user_input:
    with st.spinner("Generating answer..."):
        try:
            result = rag_chain.invoke({"input": user_input})
            answer = result.get("answer", "I don't know.")
            
            st.subheader("📘 Answer")
            st.write(answer)
            
            # Debug / Retrieved context
            with st.expander("🔍 Retrieved Knowledge Chunks"):
                for i, doc in enumerate(result.get("context", []), 1):
                    st.markdown(f"### Chunk {i}")
                    st.write(doc.page_content)
                    st.caption(f"Source: {doc.metadata.get('source', 'Unknown')}")
        
        except Exception as e:
            st.error(f"❌ Error generating answer: {str(e)}")

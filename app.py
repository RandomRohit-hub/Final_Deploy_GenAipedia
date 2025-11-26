import streamlit as st
import os

# -------------------- RAG Imports --------------------
from pinecone import Pinecone
from langchain_community.vectorstores import Pinecone as LangchainPinecone
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


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
    
    # Connect to Pinecone index using the client
    # The Index is accessed directly from the Pinecone client instance
    try:
        index = pc.Index(index_name)
    except Exception as e:
        st.error(f"Error connecting to Pinecone index: {str(e)}")
        st.info(f"Available indexes: {pc.list_indexes().names()}")
        raise
    
    # Load Pinecone vector store using langchain_community
    vectorstore = LangchainPinecone(
        index=index,
        embedding=embeddings,
        text_key="text"
    )
    
    retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
    
    # Initialize LLM
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0,
        api_key=GROQ_API_KEY
    )
    
    # Create prompt template
    template = """You are a helpful AI assistant. Use ONLY the provided context to answer.
If the answer cannot be found in the context, say: "I don't know."

Context:
{context}

Question: {question}

Answer:"""
    
    prompt = ChatPromptTemplate.from_template(template)
    
    # Build RAG chain using LCEL (LangChain Expression Language)
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain, retriever


# Initialize RAG chain
try:
    rag_chain, retriever = initialize_rag()
except Exception as e:
    st.error(f"❌ Failed to initialize RAG chain: {str(e)}")
    st.stop()


# -------------------- Chat Input --------------------
user_input = st.text_input("Enter your question:", key="user_question")

if user_input:
    with st.spinner("Generating answer..."):
        try:
            # Get answer from RAG chain
            answer = rag_chain.invoke(user_input)
            
            st.subheader("📘 Answer")
            st.write(answer)
            
            # Get retrieved documents for display
            docs = retriever.get_relevant_documents(user_input)
            
            # Debug / Retrieved context
            with st.expander("🔍 Retrieved Knowledge Chunks"):
                for i, doc in enumerate(docs, 1):
                    st.markdown(f"### Chunk {i}")
                    st.write(doc.page_content)
                    st.caption(f"Source: {doc.metadata.get('source', 'Unknown')}")
        
        except Exception as e:
            st.error(f"❌ Error generating answer: {str(e)}")

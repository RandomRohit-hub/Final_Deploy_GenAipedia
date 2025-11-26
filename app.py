import streamlit as st
import os
from pinecone import Pinecone

# Use langchain-pinecone instead of langchain-community
from langchain_pinecone import PineconeVectorStore

# Embeddings
from langchain_huggingface import HuggingFaceEmbeddings

# LLM
from langchain_groq import ChatGroq

# LangChain Core
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


# -------------------- Load Secrets --------------------
PINECONE_API_KEY = st.secrets.get("PINECONE_API_KEY", os.getenv("PINECONE_API_KEY"))
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY"))

if not PINECONE_API_KEY:
    st.error("❌ Missing PINECONE_API_KEY")
    st.stop()
if not GROQ_API_KEY:
    st.error("❌ Missing GROQ_API_KEY")
    st.stop()


# -------------------- Streamlit Setup --------------------
st.set_page_config(page_title="GenAiPedia Chatbot", layout="wide")
st.title("🤖 GenAiPedia — AI Knowledge Chatbot")
st.markdown("Ask any question related to AI/ML — grounded in your Pinecone Vector DB.")


# -------------------- Cache Embeddings --------------------
@st.cache_resource
def initialize_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )


# -------------------- Build RAG Pipeline --------------------
@st.cache_resource
def initialize_rag():
    try:
        # Initialize Pinecone
        pc = Pinecone(api_key=PINECONE_API_KEY)
        
        index_name = "genativeai-encyclopedia"
        
        # Debug info
        st.sidebar.success("🔌 Pinecone Connected")
        try:
            indexes = pc.list_indexes()
            st.sidebar.write("📊 Available Indexes:", [idx.name for idx in indexes])
        except Exception as e:
            st.sidebar.warning(f"Could not list indexes: {e}")
        
        # Initialize embeddings
        embeddings = initialize_embeddings()
        
        # Use langchain-pinecone's PineconeVectorStore (compatible with new SDK)
        vectorstore = PineconeVectorStore(
            index_name=index_name,
            embedding=embeddings,
            pinecone_api_key=PINECONE_API_KEY
        )
        
        retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
        
        # Groq LLM
        llm = ChatGroq(
            model="llama-3.1-8b-instant",
            temperature=0,
            api_key=GROQ_API_KEY
        )
        
        template = """You are a helpful AI assistant. Use ONLY the provided context to answer.
If the answer is not in the context, say: "I don't know."

Context:
{context}

Question: {question}

Answer:"""
        
        prompt = ChatPromptTemplate.from_template(template)
        
        def combine_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
        
        rag_chain = (
            {"context": retriever | combine_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        
        return rag_chain, retriever
        
    except Exception as e:
        st.error(f"❌ Initialization Error: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        st.stop()


# -------------------- Initialize System --------------------
with st.spinner("🔄 Initializing RAG pipeline..."):
    rag_chain, retriever = initialize_rag()
st.success("✅ System Ready")


# -------------------- User Input --------------------
st.markdown("---")
col1, col2 = st.columns([3, 1])

with col1:
    user_input = st.text_input(
        "🔍 Enter your question:",
        placeholder="e.g., What is supervised learning?",
        key="user_question"
    )

with col2:
    if st.button("🗑️ Clear", use_container_width=True):
        st.rerun()


# -------------------- Answer Generation --------------------
if user_input:
    with st.spinner("🤔 Thinking..."):
        try:
            answer = rag_chain.invoke(user_input)
            
            st.markdown("### 📘 Answer")
            st.write(answer)
            
            docs = retriever.get_relevant_documents(user_input)
            
            st.markdown("---")
            with st.expander(f"🔍 View {len(docs)} Retrieved Knowledge Chunks"):
                for i, doc in enumerate(docs, 1):
                    st.markdown(f"### 📄 Chunk {i}")
                    st.text_area("", doc.page_content, height=150, key=f"doc_{i}")
                    if doc.metadata:
                        st.caption(f"📎 Metadata: {doc.metadata}")
                    st.markdown("---")
        
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            import traceback
            st.code(traceback.format_exc())


# -------------------- Sidebar Info --------------------
with st.sidebar:
    st.markdown("## ℹ️ About")
    st.write("""
    - Pinecone vector DB  
    - HuggingFace Embeddings  
    - LangChain RAG pipeline  
    - Groq Llama-3.1  
    """)
    
    st.markdown("---")
    st.success("🟢 Online")
    st.info("Retrieving top 10 chunks per query")

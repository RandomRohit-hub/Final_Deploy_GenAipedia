import streamlit as st
import os

# Try different Pinecone import methods
try:
    from pinecone import Pinecone, ServerlessSpec, Index as PineconeIndex
except ImportError:
    from pinecone import Pinecone, ServerlessSpec
    PineconeIndex = None

from langchain_community.vectorstores import Pinecone as LangchainPinecone
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


# -------------------- Load Secrets --------------------
PINECONE_API_KEY = st.secrets.get("PINECONE_API_KEY", os.getenv("PINECONE_API_KEY"))
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY"))

if not PINECONE_API_KEY:
    st.error("❌ Missing PINECONE_API_KEY in Streamlit secrets")
    st.stop()
if not GROQ_API_KEY:
    st.error("❌ Missing GROQ_API_KEY in Streamlit secrets")
    st.stop()


# -------------------- Streamlit UI --------------------
st.set_page_config(page_title="GenAiPedia Chatbot", layout="wide")
st.title("🤖 GenAiPedia — AI Knowledge Chatbot")
st.markdown("Ask any question related to AI/ML — grounded in your Pinecone Vector DB.")


# -------------------- Initialize Components --------------------
@st.cache_resource
def initialize_embeddings():
    """Initialize and cache embeddings model"""
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )


@st.cache_resource
def get_pinecone_index():
    """Initialize Pinecone and get index"""
    try:
        # Initialize Pinecone client
        pc = Pinecone(api_key=PINECONE_API_KEY)
        
        # List available indexes for debugging
        try:
            indexes = pc.list_indexes()
            index_names = [idx.name for idx in indexes]
            st.sidebar.info(f"📊 Available indexes: {index_names}")
        except:
            st.sidebar.warning("Could not list indexes")
        
        # Index name
        index_name = "genativeai-encyclopedia"
        
        # Different methods to access index based on SDK version
        try:
            # Method 1: New SDK (3.x+)
            from pinecone import Index
            index = Index(index_name, api_key=PINECONE_API_KEY)
        except:
            try:
                # Method 2: Via Pinecone client
                index = pc.Index(index_name)
            except:
                # Method 3: Get index info first
                index_info = pc.describe_index(index_name)
                index = pc.Index(name=index_name, host=index_info.host)
        
        return index
    except Exception as e:
        st.error(f"Pinecone Error: {str(e)}")
        st.code(f"Pinecone module contents: {dir(pc)}")
        raise


@st.cache_resource
def initialize_rag():
    """Cache the RAG components to avoid reinitializing on every rerun"""
    
    # Get embeddings
    embeddings = initialize_embeddings()
    
    # Get Pinecone index
    index = get_pinecone_index()
    
    # Create vector store
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
    
    # Build RAG chain using LCEL
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain, retriever


# -------------------- Initialize App --------------------
try:
    with st.spinner("🔄 Initializing AI models..."):
        rag_chain, retriever = initialize_rag()
    st.success("✅ Ready to answer your questions!")
except Exception as e:
    st.error(f"❌ Failed to initialize: {str(e)}")
    st.info("💡 **Troubleshooting:**")
    st.info("1. Check if your Pinecone API key is correct")
    st.info("2. Verify index name is 'genativeai-encyclopedia'")
    st.info("3. Ensure your Pinecone index exists and is active")
    st.stop()


# -------------------- Chat Interface --------------------
st.markdown("---")

# Create columns for better layout
col1, col2 = st.columns([3, 1])

with col1:
    user_input = st.text_input("🔍 Enter your question:", placeholder="e.g., What is machine learning?", key="user_question")

with col2:
    clear_btn = st.button("🗑️ Clear", use_container_width=True)
    if clear_btn:
        st.rerun()

if user_input:
    with st.spinner("🤔 Thinking..."):
        try:
            # Get answer
            answer = rag_chain.invoke(user_input)
            
            # Display answer
            st.markdown("### 📘 Answer")
            st.markdown(answer)
            
            # Get and display source documents
            docs = retriever.get_relevant_documents(user_input)
            
            st.markdown("---")
            with st.expander(f"🔍 View {len(docs)} Retrieved Knowledge Chunks"):
                for i, doc in enumerate(docs, 1):
                    st.markdown(f"**📄 Chunk {i}**")
                    st.text_area(
                        label=f"Content {i}",
                        value=doc.page_content,
                        height=150,
                        key=f"doc_{i}",
                        label_visibility="collapsed"
                    )
                    if doc.metadata:
                        st.caption(f"📎 Metadata: {doc.metadata}")
                    st.markdown("---")
        
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            st.info("Please try rephrasing your question or contact support.")


# -------------------- Sidebar Info --------------------
with st.sidebar:
    st.markdown("## ℹ️ About")
    st.markdown("""
    This chatbot uses:
    - **Pinecone** for vector storage
    - **HuggingFace** embeddings
    - **Groq LLM** (Llama 3.1)
    - **LangChain** for RAG pipeline
    """)
    
    st.markdown("---")
    st.markdown("## 🛠️ Status")
    st.success("✅ System Online")
    st.info(f"🔢 Retrieving top 10 chunks per query")

import streamlit as st
from dotenv import load_dotenv
import os

# -------------------- RAG Imports --------------------
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

# -------------------- Load .env --------------------
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API")

if not PINECONE_API_KEY:
    st.error("❌ Missing PINECONE_API_KEY in environment")
if not GROQ_API_KEY:
    st.error("❌ Missing GROQ_API in environment")


# -------------------- Pinecone Initialization --------------------
pc = Pinecone(api_key=PINECONE_API_KEY)

index_name = "genativeai-encyclopedia"
namespace = ""   # change if you used a namespace when inserting vectors

# Check index exists
if index_name not in pc.list_indexes().names():
    st.error(f"❌ Pinecone index '{index_name}' does not exist!")
else:
    st.success(f"Connected to Pinecone index: {index_name}")


# -------------------- Streamlit UI --------------------
st.set_page_config(page_title="GenAiPedia Chatbot", layout="wide")
st.title("🤖 GenAiPedia — AI Knowledge Chatbot")
st.markdown("Ask any question related to AI/ML. Answers are grounded in your Pinecone vector DB.")


# -------------------- Embeddings --------------------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"}
)


# -------------------- Vector Store --------------------
db = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings,
    namespace=namespace
)

retriever = db.as_retriever(search_kwargs={"k": 10})


# -------------------- Groq LLM --------------------
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
    api_key=GROQ_API_KEY
)


# -------------------- Prompt Template --------------------
system_prompt = """
You are a helpful and knowledgeable AI assistant. 
Use ONLY the context provided below to answer the question.
If the answer is not in the context, say: "I don't know."

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


# -------------------- Chat Input --------------------
user_input = st.text_input("Enter your question:")

if user_input:
    with st.spinner("Generating answer..."):
        result = rag_chain.invoke({"input": user_input})
        answer = result.get("answer", "I don't know.")

    st.subheader("📘 Answer")
    st.write(answer)

    # Show retrieved context chunks
    with st.expander("🔍 Retrieved Knowledge Chunks"):
        for i, doc in enumerate(result.get("context", []), start=1):
            st.markdown(f"### Chunk {i}")
            st.write(doc.page_content)
            st.caption(f"Source: {doc.metadata.get('source')}")

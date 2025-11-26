# ----------------- LOAD ENVIRONMENT VARIABLES -----------------
from dotenv import load_dotenv
import os

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API")   # matches .env exactly

# ✅ Validate keys
if not PINECONE_API_KEY:
    raise ValueError("❌ PINECONE_API_KEY is missing in .env")

if not GROQ_API_KEY:
    raise ValueError("❌ GROQ_API is missing in .env")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GROQ_API"] = GROQ_API_KEY

print("✅ API keys loaded successfully")


# ----------------- SETUP PINECONE INDEX -----------------
from pinecone import Pinecone, ServerlessSpec

pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "genativeai-encyclopedia"   # must be lowercase

existing_indexes = [i["name"] for i in pc.list_indexes()]

if index_name not in existing_indexes:
    print("🆕 Creating Pinecone index...")
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
else:
    print("✅ Index already exists")

index = pc.Index(index_name)
print("✅ Pinecone index ready:", index_name)


# ----------------- LOAD DOCUMENTS -----------------
from typing import List
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import DirectoryLoader, PyMuPDFLoader

DATA_PATH = r"D:\XXX.BADI BAAT CHEET.XXX\GenAiPedia\GenAIPed\data"  # ✅ Correct absolute path

loader = DirectoryLoader(
    DATA_PATH,
    glob="*.pdf",
    loader_cls=PyMuPDFLoader
)

documents = loader.load()
print(f"📄 Loaded {len(documents)} documents from: {DATA_PATH}")


# ----------------- CLEAN & SPLIT -----------------
def filter_to_minimal(docs: List[Document]) -> List[Document]:
    cleaned = []
    for doc in docs:
        cleaned.append(
            Document(
                page_content=doc.page_content,
                metadata={"source": doc.metadata.get("source", "unknown")}
            )
        )
    return cleaned

minimal_docs = filter_to_minimal(documents)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150
)
text_chunks = splitter.split_documents(minimal_docs)

print(f"✂️ Split into {len(text_chunks)} text chunks")


# ----------------- EMBEDDINGS -----------------
from langchain.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"}
)

print("🧠 Embeddings loaded (CPU mode)")


# ----------------- STORE IN PINECONE -----------------
from langchain_pinecone import PineconeVectorStore

docsearch = PineconeVectorStore.from_documents(
    documents=text_chunks,
    embedding=embeddings,
    index_name=index_name
)

print(" Successfully indexed all documents into Pinecone!")

from langchain.document_loaders  import PyMuPDFLoader,DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader

def load_pdf_data(folder_path):
    loader = DirectoryLoader(
        folder_path,
        glob="*.pdf",
        loader_cls=PyMuPDFLoader
    )
    return loader.load()



from typing import List
from langchain.schema import Document

def filter_to_minimal(docs: List[Document]) -> List[Document]:
    """
    Given a list of Document objects, return a new list containing
    only `page_content` and the `source` field in metadata.
    """

    minimal_d: List[Document] = []
    for doc in docs:  # <- use doc instead of docs
        src = doc.metadata.get("source")
        minimal_d.append(
            Document(
                page_content=doc.page_content,
                metadata={'source': src}  # fixed typo
            )
        )

    return minimal_d




# splitting into smaller chunks

def tect_splitter(minimal_d):
    text_splitter=RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    text_chunks=text_splitter.split_documents(minimal_d)
    return text_chunks



import torch
from langchain.embeddings import HuggingFaceEmbeddings

def download_embedding():
    """
    Download and return HuggingFace embeddings
    """

    model_name = "sentence-transformers/all-MiniLM-L6-v2"  
    device = "cuda" if torch.cuda.is_available() else "cpu"

    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={'device': device}
    )

    return embeddings

embeddings = download_embedding()
print("✅ Embedding model loaded on:", "GPU" if torch.cuda.is_available() else "CPU")



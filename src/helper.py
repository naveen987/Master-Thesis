from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings

# Extract data from the PDF
def load_pdf(data):
    """
    Load PDF files from the specified directory and extract their content.
    """
    loader = DirectoryLoader(data, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    return documents

# Create text chunks
def text_split(extracted_data):
    """
    Split the extracted PDF content into smaller text chunks for processing.
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    text_chunks = text_splitter.split_documents(extracted_data)
    return text_chunks

# Download 768-dimensional embedding model
def download_hugging_face_embeddings():
    """
    Load a 768-dimensional embedding model compatible with the Pinecone index.
    """
    model_name = "sentence-transformers/all-mpnet-base-v2"  # 768-dimensional model
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    return embeddings

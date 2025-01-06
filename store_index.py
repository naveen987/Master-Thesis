from src.helper import load_pdf, text_split, download_hugging_face_embeddings
from langchain.vectorstores import Pinecone
from pinecone import Pinecone as PineconeClient, ServerlessSpec
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
PINECONE_API_ENV = os.environ.get('PINECONE_API_ENV')

# Load PDF data and split into chunks
extracted_data = load_pdf("data/")
text_chunks = text_split(extracted_data)

# Download embeddings
embeddings = download_hugging_face_embeddings()

# Initialize Pinecone client
pinecone_instance = PineconeClient(api_key=PINECONE_API_KEY)

# Index name
index_name = "new"

# Check if the index exists, create it if not
if index_name not in [index.name for index in pinecone_instance.list_indexes()]:
    pinecone_instance.create_index(
        name=index_name,
        dimension=768,  # Update this dimension to match your embedding model
        metric="cosine",  # Use 'cosine', 'euclidean', or other metrics as needed
        spec=ServerlessSpec(
            cloud="aws",  # Update the cloud provider if different
            region=PINECONE_API_ENV
        )
    )

# Access the index
docsearch = Pinecone.from_texts(
    [t.page_content for t in text_chunks],
    embeddings,
    index_name=index_name
)

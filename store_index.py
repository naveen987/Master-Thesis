import json
from src.helper import download_hugging_face_embeddings
from langchain.vectorstores import Pinecone
from pinecone import Pinecone as PineconeClient, ServerlessSpec
from dotenv import load_dotenv
import os

# Load environment variablesmm
load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
PINECONE_API_ENV = os.environ.get('PINECONE_API_ENV')

# Load JSON data
with open('scraped_data_subsections.json', 'r') as file:
    data = json.load(file)

# Extract text chunks from JSON data
text_chunks = []
for item in data:
    for subsection in item['subsections']:
        # Combine heading and content for meaningful context
        heading = subsection.get('heading', '')
        content = ' '.join(subsection.get('content', []))
        text_chunks.append(f"{heading}: {content}")

# Download embeddings
embeddings = download_hugging_face_embeddings()

# Initialize Pinecone client
pinecone_instance = PineconeClient(api_key=PINECONE_API_KEY)

# Index name
index_name = "zf"

# Check if the index exists, create it if not
if index_name not in [index.name for index in pinecone_instance.list_indexes()]:
    pinecone_instance.create_index(
        name=index_name,
        dimension=768,  # Update this dimension to match your embedding model
        metric="cosine",  # Use 'cosine', 'euclidean', or other metrics as needed
        spec=ServerlessSpec(
            cloud="aws",  # Update the cloud provider if different
            region="us-east-1"
        )
    )

# Access the index and upload data
docsearch = Pinecone.from_texts(
    text_chunks,  # Use the prepared text chunks
    embeddings,
    index_name=index_name
)

print("Data successfully uploaded to Pinecone!")

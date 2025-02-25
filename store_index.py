import json
from src.helper import download_hugging_face_embeddings
from langchain.vectorstores import Pinecone
from pinecone import Pinecone as PineconeClient, ServerlessSpec
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Retrieve Pinecone API credentials from environment variables
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
PINECONE_API_ENV = os.environ.get('PINECONE_API_ENV')

# Load JSON data from the first file
with open('scraped_data_subsections_c.json', 'r', encoding='utf-8') as file1:
    data1 = json.load(file1)

# Extract text chunks from the first JSON
text_chunks = []
for item in data1:
    for subsection in item['subsections']:
        heading = subsection.get('heading', '')
        content = ' '.join(subsection.get('content', []))
        text_chunks.append(f"{heading}: {content}")

# Load JSON data from the second file
#with open('cleaned_doctors_data_multiple_locations.json', 'r', encoding='utf-8') as file2:
    #data2 = json.load(file2)

# Extract text chunks from the second JSON
#for entry in data2:
#    doctor_name = entry.get('Doctor Name', entry.get('doctor_name', 'Unknown Doctor'))
#    specialty = entry.get('Specialty', entry.get('specialty', 'General'))
#    location = entry.get('Location', entry.get('location', 'Unknown Location'))
#    address = entry.get('Address', entry.get('address', 'Unknown Address'))
#    text_chunks.append(f"Doctor: {doctor_name}, Specialty: {specialty}, Location: {location}, Address: {address}")

# Download embeddings
embeddings = download_hugging_face_embeddings()

# Initialize Pinecone client
pinecone_instance = PineconeClient(api_key=PINECONE_API_KEY)

# Index name
index_name = "zf"

# Check if the index exists, create it if not
if index_name not in [index.name for index in pinecone_instance.list_indexes()]:
    print(f"Creating index '{index_name}'...")
    pinecone_instance.create_index(
        name=index_name,
        dimension=768,  # Ensure this matches your embedding model's output dimension
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region=PINECONE_API_ENV)
    )

# Upload data to Pinecone index
docsearch = Pinecone.from_texts(
    text_chunks,
    embeddings,
    index_name=index_name
)

print("Data successfully uploaded to Pinecone!")

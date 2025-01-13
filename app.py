import chainlit as cl
from src.helper import download_hugging_face_embeddings
from langchain_community.vectorstores import Pinecone
from pinecone import Pinecone as PineconeClient, ServerlessSpec
from langchain.prompts import PromptTemplate
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import os
import logging
import warnings
import asyncio

# Suppress PyTorch warnings
warnings.filterwarnings("ignore", message=".*torch.utils._pytree.*")

# Logging setup
logging.basicConfig(level=logging.INFO)

# Load environment variables
load_dotenv()
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_API_ENV = os.environ.get("PINECONE_API_ENV")
index_name = "zf"

# Load embeddings for all-mpnet-base-v2 (768-dimensional embeddings)
embedding_model = SentenceTransformer('all-mpnet-base-v2')
embeddings = download_hugging_face_embeddings()

# Initialize Pinecone
pinecone_instance = PineconeClient(api_key=PINECONE_API_KEY)
if index_name not in [index.name for index in pinecone_instance.list_indexes()]:
    pinecone_instance.create_index(
        name=index_name,
        dimension=768,  # Updated for 768-dimensional embeddings
        metric='cosine',
        spec=ServerlessSpec(cloud='aws', region=PINECONE_API_ENV)
    )

# Access the index
docsearch = Pinecone.from_existing_index(index_name, embeddings)

# Define prompt template
PROMPT = PromptTemplate(
    template=(
        "You are a medical assistant. Based on the following context, "
        "answer the question concisely and accurately:\n\n"
        "Context: {context}\n\n"
        "Question: {question}\n\n"
        "Answer:"
    ),
    input_variables=["context", "question"]
)
chain_type_kwargs = {"prompt": PROMPT}

# Define Chat Profiles
@cl.set_chat_profiles
async def chat_profiles():
    return [
        cl.ChatProfile(
            name="Mistral Biomedical",
            markdown_description="An assistant specialized in biomedical and medical queries.",
        )
    ]

# Initialize the LLM
def initialize_llm(profile_name):
    model_path = "model/mistral-13b-v0.1.Q3_K_M.gguf"  # Updated to Mistral Long Context model path
    return CTransformers(
        model=model_path,
        model_type="mistral",
        config={
            'max_new_tokens': 4096,  # Number of tokens to generate
            'context_length': 4096,  # Explicitly set context length to 4096
            'temperature': 0.5
        }
    )

# Truncate context for the model's token limit
def truncate_context(context, max_tokens=4096):
    """
    Truncate the context to fit within the maximum token limit.
    """
    tokens = context.split()
    if len(tokens) > max_tokens:
        return " ".join(tokens[-max_tokens:])  # Keep only the last max_tokens tokens
    return context

# Define synchronous long-running task
def long_running_task(input_data, llm):
    """Improved long-running synchronous task for LLM inference."""
    logging.info("Generating response...")
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=docsearch.as_retriever(search_kwargs={'k': 5}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    result = qa.invoke(input_data)
    logging.info("Response generated: %s", result["result"])
    return result["result"]

# Convert to async task
async_long_running_task = cl.make_async(long_running_task)

@cl.on_chat_start
async def start_chat():
    """Send a welcome message when the chat starts."""
    await cl.Message(content="Welcome! Upload a PDF or ask a biomedical question to get started.").send()

@cl.on_message
async def handle_message(message):
    """Handle user messages and file uploads."""
    try:
        if message.type == "file":
            logging.info("Processing uploaded file...")
            loader = PyPDFLoader(message.file_path)
            documents = loader.load()

            # Split text into manageable chunks
            text_splitter = CharacterTextSplitter(chunk_size=4000, chunk_overlap=500)
            texts = text_splitter.split_documents(documents)

            # Add chunks to Pinecone index in batches
            batch_size = 10
            for i in range(0, len(texts), batch_size):
                docsearch.add_texts([t.page_content for t in texts[i:i + batch_size]])

            await cl.Message(content="PDF uploaded and processed successfully! You can now ask questions about its content.").send()
            return

        # Handle regular text messages
        logging.info("Message received from UI: %s", message.content)
        query = message.content

        # Retrieve the selected chat profile
        chat_profile = cl.user_session.get("chat_profile", "Mistral Biomedical")

        # Initialize the LLM based on the selected chat profile
        llm = initialize_llm(chat_profile)

        # Send a processing message
        await cl.Message(content=f"Processing your request with the '{chat_profile}' profile, please wait...").send()

        # Retrieve documents
        retriever = docsearch.as_retriever(search_kwargs={'k': 5})
        docs = retriever.invoke(query)

        # Combine and truncate context
        context = " ".join([doc.page_content for doc in docs])
        truncated_context = truncate_context(context, max_tokens=4096)

        # Prepare input data for the LLM
        input_data = {"query": query, "context": truncated_context}

        # Call the async long-running task
        result = await async_long_running_task(input_data, llm)

        # Update the user with the result
        await cl.Message(content=result).send()
        logging.info("Response sent successfully!")
    except Exception as e:
        logging.error("Error occurred: %s", e)
        await cl.Message(content="An error occurred while processing your request. Please try again.").send()

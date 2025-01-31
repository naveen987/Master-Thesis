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
from twilio.rest import Client  # Twilio for SMS
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
TWILIO_ACCOUNT_SID = os.environ.get("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.environ.get("TWILIO_AUTH_TOKEN")
TWILIO_PHONE_NUMBER = os.environ.get("TWILIO_PHONE_NUMBER")

index_name = "zf"

# Load embeddings for all-mpnet-base-v2 (768-dimensional embeddings)
embedding_model = SentenceTransformer('all-mpnet-base-v2')
embeddings = download_hugging_face_embeddings()

# Initialize Pinecone
pinecone_instance = PineconeClient(api_key=PINECONE_API_KEY)
if index_name not in [index.name for index in pinecone_instance.list_indexes()]:
    pinecone_instance.create_index(
        name=index_name,
        dimension=768,
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

# Twilio Client for SMS
twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

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
def initialize_llm():
    model_path = "model/mistral-13b-v0.1.Q3_K_M.gguf"
    return CTransformers(
        model=model_path,
        model_type="mistral",
        config={
            'max_new_tokens': 4096,
            'context_length': 4096,
            'temperature': 0.5
        }
    )

llm = initialize_llm()

# Initialize user_appointment_info
global user_appointment_info
user_appointment_info = {}

def long_running_task(input_data, llm):
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

async_long_running_task = cl.make_async(long_running_task)

@cl.on_chat_start
async def start_chat():
    await cl.Message(content="Welcome! Upload a PDF or ask a biomedical question to get started. To book an appointment, type 'book an appointment'.").send()

@cl.on_message
async def handle_message(message):
    try:
        global user_appointment_info
        if message.type == "file":
            logging.info("Processing uploaded file...")
            loader = PyPDFLoader(message.file_path)
            documents = loader.load()

            text_splitter = CharacterTextSplitter(chunk_size=4000, chunk_overlap=500)
            texts = text_splitter.split_documents(documents)

            batch_size = 10
            for i in range(0, len(texts), batch_size):
                docsearch.add_texts([t.page_content for t in texts[i:i + batch_size]])

            await cl.Message(content="PDF uploaded and processed successfully! You can now ask questions about its content.").send()
            return

        query = message.content.lower()

        # Check if the user wants to book an appointment
        if "book an appointment" in query:
            user_appointment_info[message.author] = {}
            await cl.Message(content="Sure! What is your location (e.g., Berlin, Munich)?").send()
            return

        if message.author in user_appointment_info:
            if "location" not in user_appointment_info[message.author]:
                user_appointment_info[message.author]["location"] = message.content
                await cl.Message(content="Got it! What specialty do you need (e.g., cardiology, dermatology)?").send()
                return
            elif "specialty" not in user_appointment_info[message.author]:
                user_appointment_info[message.author]["specialty"] = message.content
                await cl.Message(content="Finally, please provide your phone number so we can send the booking link.").send()
                return
            elif "phone_number" not in user_appointment_info[message.author]:
                user_appointment_info[message.author]["phone_number"] = message.content
                location = user_appointment_info[message.author]["location"].replace(" ", "-")
                specialty = user_appointment_info[message.author]["specialty"].replace(" ", "-")
                booking_link = f"https://www.doctolib.de/{specialty}/{location}"
                twilio_client.messages.create(
                    body=f"Your appointment booking link: {booking_link}",
                    from_=TWILIO_PHONE_NUMBER,
                    to=user_appointment_info[message.author]["phone_number"]
                )
                await cl.Message(content=f"Your booking link has been sent to {message.content}.").send()
                del user_appointment_info[message.author]
                return

        chat_profile = cl.user_session.get("chat_profile", "Mistral Biomedical")
        llm = initialize_llm()
        await cl.Message(content=f"Processing your request with the '{chat_profile}' profile, please wait...").send()
        retriever = docsearch.as_retriever(search_kwargs={'k': 5})
        docs = retriever.invoke(query)
        context = " ".join([doc.page_content for doc in docs])
        input_data = {"query": query, "context": context}
        result = await async_long_running_task(input_data, llm)
        await cl.Message(content=result).send()
    except Exception as e:
        logging.error("Error occurred: %s", e)
        await cl.Message(content="An error occurred while processing your request. Please try again.").send()
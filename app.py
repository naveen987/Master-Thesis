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

# Specialty translation mapping (English -> German)
SPECIALTY_TRANSLATIONS = {
    "cardiology": "kardiologie",
    "dermatology": "dermatologie",
    "neurology": "neurologie",
    "orthopedics": "orthopadie",
    "pediatrics": "padiatrie",
    "gynecology": "gynakologie",
    "urology": "urologie",
    "endocrinology": "endokrinologie",
    "gastroenterology": "gastroenterologie",
    "ophthalmology": "augenheilkunde",
    "psychiatry": "psychiatrie",
    "pulmonology": "pneumologie",
    "rheumatology": "rheumatologie",
    "dentistry": "zahnarzt",
    "general practice": "hausarzt",
    "otolaryngology (ent)": "hals-nasen-ohren-heilkunde",
    "physical therapy": "physiotherapie",
    "radiology": "radiologie",
    "oncology": "onkologie",
    "hematology": "hamatologie",
    "anesthesiology": "anästhesiologie",
    "nephrology": "nephrologie",
    "surgery": "chirurgie",
    "internal medicine": "innere-medizin",
    "infectious diseases": "infektionskrankheiten",
    "sports medicine": "sportmedizin",
    "immunology": "immunologie",
    "nutrition": "ernährungsberatung",
    "occupational medicine": "arbeitsmedizin",
    "pain management": "schmerztherapie",
}

def translate_specialty(english_specialty):
    """Convert English medical specialty to its German equivalent."""
    specialty_lower = english_specialty.lower().strip()
    return SPECIALTY_TRANSLATIONS.get(specialty_lower, specialty_lower)  # Fallback to original if not found

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

# Store user input for appointment booking
user_appointment_info = {}

@cl.on_chat_start
async def start_chat():
    await cl.Message(content="Welcome! Upload a PDF or ask a biomedical question to get started.").send()

@cl.on_message
async def handle_message(message):
    """Handles user messages and appointment booking."""
    try:
        global user_appointment_info

        # Step 1: Check if user wants to book an appointment
        if "book an appointment" in message.content.lower():
            user_appointment_info[message.author] = {}
            await cl.Message(content="Sure! What is your location (e.g., Berlin, Munich)?").send()
            return

        # Step 2: Get the location
        if message.author in user_appointment_info and "location" not in user_appointment_info[message.author]:
            user_appointment_info[message.author]["location"] = message.content.lower()
            await cl.Message(content="Got it! What specialty do you need (e.g., cardiology, dermatology)?").send()
            return

        # Step 3: Get the medical specialty and translate it
        if message.author in user_appointment_info and "specialty" not in user_appointment_info[message.author]:
            english_specialty = message.content.lower().strip()
            german_specialty = translate_specialty(english_specialty)  # Ensure translation happens
            user_appointment_info[message.author]["specialty"] = german_specialty
            await cl.Message(content="Finally, please provide your phone number so we can send the booking link.").send()
            return

        # Step 4: Get the phone number and send booking link
        if message.author in user_appointment_info and "phone_number" not in user_appointment_info[message.author]:
            user_appointment_info[message.author]["phone_number"] = message.content

            # Ensure translated specialty is used
            location = user_appointment_info[message.author]["location"].replace(" ", "-")
            specialty = user_appointment_info[message.author]["specialty"].replace(" ", "-")

            # Correct Doctolib link with German specialty
            booking_link = f"https://www.doctolib.de/{specialty}/{location}"

            # Send SMS via Twilio
            twilio_client.messages.create(
                body=f"Your appointment booking link: {booking_link}",
                from_=TWILIO_PHONE_NUMBER,
                to=user_appointment_info[message.author]["phone_number"]
            )

            # Confirm to the user
            await cl.Message(content=f"Your booking link has been sent to {message.content}.").send()
            
            # Clear stored data
            del user_appointment_info[message.author]
            return

    except Exception as e:
        logging.error("Error occurred: %s", e)
        await cl.Message(content="An error occurred while processing your request. Please try again.").send()

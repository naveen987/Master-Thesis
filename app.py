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

# -- DB reading for user roles
from database import get_user  # Make sure you have a get_user() function

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

# 1) Load embeddings for all-mpnet-base-v2 (768-dimensional embeddings)
embedding_model = SentenceTransformer('all-mpnet-base-v2')
embeddings = download_hugging_face_embeddings()

# 2) Initialize Pinecone
pinecone_instance = PineconeClient(api_key=PINECONE_API_KEY)
if index_name not in [index.name for index in pinecone_instance.list_indexes()]:
    pinecone_instance.create_index(
        name=index_name,
        dimension=768,
        metric='cosine',
        spec=ServerlessSpec(cloud='aws', region=PINECONE_API_ENV)
    )

# 3) Access the index
docsearch = Pinecone.from_existing_index(index_name, embeddings)

# 4) Define prompt template
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

# 5) Twilio Client for SMS (optional)
twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

# 6) Define Chat Profiles
@cl.set_chat_profiles
async def chat_profiles():
    return [
        cl.ChatProfile(
            name="Mistral Biomedical",
            markdown_description="An assistant specialized in biomedical and medical queries.",
        )
    ]

# 7) Initialize the LLM
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

# 8) Appointment info global
global user_appointment_info
user_appointment_info = {}

# 9) Long-running QA task
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

# 10) Chat start logic
@cl.on_chat_start
async def start_chat():
    """
    On chat start, we read LOGGED_IN_EMAIL from environment, 
    fetch the user's role from DB, store it in session, and greet them.
    """
    email = os.environ.get("LOGGED_IN_EMAIL", "")
    role = None

    if email:
        user = get_user(email)  # e.g. (id, email, password, role)
        if user:
            role = user[3].strip().upper()  # e.g. 'PATIENT' or 'STUDENT'

    cl.user_session.set("role", role)

    if role == "PATIENT":
        await cl.Message(
            content=(
                "🩺 **Patient Chatbot**\n\n"
                "Welcome! Ask a biomedical question or Provide Your Symptoms. "
                "To book an appointment, type 'book an appointment'."
            )
        ).send()

    elif role == "STUDENT":
        await cl.Message(
            content=(
                "📘 **Student Chatbot**\n\n"
                "Welcome! You can upload a PDF (paperclip icon) or ask a biomedical question to get started."
            )
        ).send()

    else:
        await cl.Message(content="❌ **ERROR: Role not recognized. Contact Support.**").send()

# 11) Main message handler
@cl.on_message
async def handle_message(message):
    try:
        global user_appointment_info
        role = cl.user_session.get("role")  # 'PATIENT' or 'STUDENT' or None

        # If user uploaded a file via paperclip
        if message.type == "file":
            logging.info("Processing uploaded file from user via paperclip...")
            loader = PyPDFLoader(message.file_path)
            documents = loader.load()

            text_splitter = CharacterTextSplitter(chunk_size=4000, chunk_overlap=500)
            texts = text_splitter.split_documents(documents)

            batch_size = 10
            for i in range(0, len(texts), batch_size):
                docsearch.add_texts([t.page_content for t in texts[i : i + batch_size]])

            await cl.Message(
                content=(
                    "✅ PDF uploaded and processed successfully! "
                    "You can now ask questions about its content."
                )
            ).send()
            return

        # Otherwise, treat the message as text
        query = message.content.lower()

        # Appointment logic for PATIENT
        if role == "PATIENT":
            if "book an appointment" in query:
                user_appointment_info[message.author] = {}
                await cl.Message(
                    content="📍 Sure! What is your location (e.g., Berlin, Munich)?"
                ).send()
                return

            if message.author in user_appointment_info:
                if "location" not in user_appointment_info[message.author]:
                    user_appointment_info[message.author]["location"] = message.content
                    await cl.Message(
                        content="🩺 Got it! What specialty do you need (e.g., cardiology, dermatology, neurology, etc.)?"
                    ).send()
                    return
                elif "specialty" not in user_appointment_info[message.author]:
                    user_appointment_info[message.author]["specialty"] = message.content
                    await cl.Message(
                        content="📞 Finally, please provide your phone number so we can send the booking link."
                    ).send()
                    return
                elif "phone_number" not in user_appointment_info[message.author]:
                    user_appointment_info[message.author]["phone_number"] = message.content
                    
                    # Mandatory conversion of city to German using a comprehensive translation map
                    city_raw = user_appointment_info[message.author]["location"].strip()
                    city_translation_map = {
                        "berlin": "berlin",
                        "munich": "muenchen",
                        "cologne": "koeln",
                        "nuremberg": "nuernberg",
                        "wurzburg": "wurzburg",
                        "frankfurt": "frankfurt-am-main",
                        "stuttgart": "stuttgart",
                        "dusseldorf": "duesseldorf",
                        "hanover": "hannover",
                        "hamburg": "hamburg",
                        "leipzig": "leipzig",
                        "dresden": "dresden",
                        "bremen": "bremen",
                        "dortmund": "dortmund",
                        "essen": "essen",
                        "bielefeld": "bielefeld",
                        "bonn": "bonn",
                        "mannheim": "mannheim",
                        "karlsruhe": "karlsruhe",
                        "wiesbaden": "wiesbaden",
                        "augsburg": "augsburg",
                        "aachen": "aachen",
                        "osnabrueck": "osnabrueck",
                        "darmstadt": "darmstadt",
                        "regensburg": "regensburg",
                        "ingolstadt": "ingolstadt",
                        "wolfsburg": "wolfsburg",
                        "ulm": "ulm",
                        "offenbach": "offenbach",
                        "heilbronn": "heilbronn",
                        "pforzheim": "pforzheim",
                        "oldenburg": "oldenburg",
                        "kassel": "kassel",
                        "fuerth": "fuerth",
                        "magdeburg": "magdeburg",
                        "freiburg": "freiburg",
                        "kiel": "kiel",
                        "rostock": "rostock",
                        "chemnitz": "chemnitz",
                        "hagen": "hagen",
                        "saarbruecken": "saarbruecken",
                        "potsdam": "potsdam",
                        "ludwigshafen": "ludwigshafen-am-rhein",
                        "witten": "witten",
                        "reutlingen": "reutlingen",
                        "koblenz": "koblenz",
                        "bremerhaven": "bremerhaven",
                        "solingen": "solingen",
                        "heidelberg": "heidelberg"
                    }
                    city_lower = city_raw.lower()
                    city_converted = city_translation_map.get(city_lower, city_raw)
                    city_converted = city_converted.replace(" ", "-")
                    
                    # Mandatory conversion of specialty to German with a comprehensive translation map
                    specialty_raw = user_appointment_info[message.author]["specialty"].strip()
                    translation_map = {
                        "cardiology": "kardiologie",
                        "dermatology": "dermatologie",
                        "neurology": "neurologie",
                        "pediatrics": "pädiatrie",
                        "orthopedics": "orthopädie",
                        "gastroenterology": "gastroenterologie",
                        "endocrinology": "endokrinologie",
                        "rheumatology": "rheumatologie",
                        "pulmonology": "pneumologe",
                        "oncology": "onkologie",
                        "urology": "urologie",
                        "gynecology": "gynäkologie",
                        "ophthalmology": "ophthalmologie",
                        "otolaryngology": "hals-nasen-ohren-heilkunde",
                        "psychiatry": "psychiatrie",
                        "surgery": "chirurgie",
                        "radiology": "radiologie",
                        "nephrology": "nephrologie",
                        "immunology": "immunologie",
                        "allergy": "allergologie"
                    }
                    specialty_lower = specialty_raw.lower()
                    specialty_converted = translation_map.get(specialty_lower, specialty_raw)
                    specialty_converted = specialty_converted.replace(" ", "-")
                    
                    booking_link = f"https://www.doctolib.de/{specialty_converted}/{city_converted}"
                    twilio_client.messages.create(
                        body=f"📅 Your appointment booking link: {booking_link}",
                        from_=TWILIO_PHONE_NUMBER,
                        to=user_appointment_info[message.author]["phone_number"]
                    )
                    await cl.Message(
                        content=f"✅ Your booking link has been sent to {message.content}."
                    ).send()
                    del user_appointment_info[message.author]
                    return

        # Common logic for STUDENT and PATIENT
        chat_profile = cl.user_session.get("chat_profile", "Mistral Biomedical")
        llm = initialize_llm()

        # Let user know we are processing
        await cl.Message(
            content=f"⏳ Processing your request with the **'{chat_profile}'** profile, please wait..."
        ).send()

        retriever = docsearch.as_retriever(search_kwargs={'k': 5})
        docs = retriever.invoke(query)
        context = " ".join([doc.page_content for doc in docs])

        input_data = {"query": query, "context": context}
        result = await async_long_running_task(input_data, llm)

        await cl.Message(content=result).send()

    except Exception as e:
        logging.error(f"❌ Error occurred: {e}")
        await cl.Message(
            content="⚠️ An error occurred while processing your request. Please try again."
        ).send()

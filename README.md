# SRH-CHATBOT-V3

SRH-CHATBOT-V3 is a sophisticated chatbot system tailored for university environments, designed to deliver efficient and precise responses by utilizing cutting-edge technologies such as Meta Llama2 and Pinecone. Engineered for simplicity and scalability, it is ideally suited for diverse university-related applications, from streamlining administrative tasks and improving operational efficiency to facilitating interactive learning environments for students and faculty. This chatbot aims to transform communication within the university, ensuring that information and academic support are more accessible and effective for the entire university community.

## Getting Started
Follow these steps to get your copy of SRH-CHATBOT-V3 up and running on your local machine for development and testing purposes.

### Prerequisites
- **Anaconda**: Download and install Anaconda from the [official Anaconda website](https://www.anaconda.com/products/individual).
- **Pinecone Account**: Sign up at [Pinecone.io](https://www.pinecone.io/).
- **Google Cloud Storage Bucket**: Create a bucket in your Google Cloud account.
- **Bing Search API Key**: Obtain your API key by following the steps at [Create a Bing Search Service Resource](https://docs.microsoft.com/en-us/azure/cognitive-services/bing-web-search/).

### Setup Instructions

#### Step 1: Clone the Repository
Clone the SRH-CHATBOT-V3 repository to your local machine using the command:
```bash
git clone https://github.com/hassanspaceimam/SRH-CHATBOT-V3.git
```

#### Step 2: Create a Conda Environment
Navigate to the project directory and create a new Conda environment named `schatbot` with Python 3.8:
```bash
conda create -n schatbot python=3.8 -y
```
Activate the environment:
```bash
conda activate schatbot
```

#### Step 3: Install Dependencies
Install the required Python packages listed in `requirements.txt`:
```bash
pip install -r requirements.txt
```

#### Step 4: Set Up Environment Variables
Create a `.env` file in the root directory and add your Pinecone credentials and Bing Search API key:
```ini
PINECONE_API_KEY = "your_pinecone_api_key_here"
PINECONE_API_ENV = "your_pinecone_environment_here"
BING_SEARCH_API_KEY = "your_bing_search_api_key_here"
```

#### Step 5: Download and Set Up the Model
Download the Llama 2 Model llama-2-13b-chat.ggmlv3.q4_0.bin from [Hugging Face](https://huggingface.co/TheBloke/Llama-2-13B-chat-GGML) and place it in the model directory.


#### Step 6: Prepare Data Index
Before proceeding with the data index preparation, ensure that you have created a Pinecone index named `srh-heidelberg-docs` with the following specifications:
- **Metric**: Cosine
- **Dimensions**: 768

This Pinecone index is essential for storing and retrieving vectorized representations of your documents. Once the index is set up, choose one of the following methods to prepare and upload your data:

- **Using Local PDF Documents**:
  If you have PDF documents available locally in your `data` folder, you can upload their vectors to the Pinecone index. Use the `store_index.py` script to convert your local PDF files into vectors and upload them:
  ```bash
  python store_index.py
  ```
- **Using PDF Documents Available in a Google Cloud Bucket**:
  If you prefer to load your PDF documents from a Google Cloud Storage bucket, update the BUCKET_NAME in the gcs_store_index.py script with your actual bucket name. Specify the absolute path to your data folder in the DATA_FOLDER variable within the script. This method will download the PDF documents from your Google Cloud Storage bucket, convert them to vectors, and upload them to the Pinecone index:
  ```bash
  python gcs_store_index.py
  ```


### Launch the Application

#### Using Streamlit
Run the application using Streamlit:
```bash
streamlit run appstreamlit.py
```
Access the web interface at `http://localhost:8501/`.

#### Using Flask with Customizable UI
For a more personalized user interface, Flask allows you to customize the UI using HTML and CSS. The template and stylesheet for customization can be found at the following locations within the SRH-CHATBOT-V3 project:

- **HTML Template**: `SRH-CHATBOT-V3/templates/chat.html` - This file serves as the layout for the chat interface. You can edit this file to modify the structure, elements, and overall design of the web page.
- **CSS Stylesheet**: `SRH-CHATBOT-V3/static/style.css` - This stylesheet contains the style rules for the web interface. Customize this file to change the look and feel of the chatbot UI, including colors, fonts, spacing, and more.

After making your desired customizations to the HTML template and CSS stylesheet, launch the application with Flask by running:
```bash
python app.py
```
Access the web interface at `http://localhost:8080/`.

### Tech Stack:
- Python: Core programming language
- LangChain: Library for building language model applications
- Flask: Web framework for building the backend
- Meta Llama2: AI model for natural language understanding and generation
- Pinecone: Vector database for similarity search
- Streamlit: Framework for building interactive web apps
- Google Cloud Storage: Cloud storage for PDF files
- Bing Search API: Web search capabilities

### Acknowledgments
Special thanks to SRH Heidelberg for supporting this project.

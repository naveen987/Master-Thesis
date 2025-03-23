import os
import time
import logging
import numpy as np
import plotly.graph_objects as go
from sentence_transformers import SentenceTransformer, util

# Import your LLM chain initialization and retrieval functions.
from app import initialize_llm, long_running_task, docsearch, PROMPT

# Set up logging
logging.basicConfig(level=logging.INFO)

# Load a SentenceTransformer model for semantic similarity evaluation.
similarity_model = SentenceTransformer('all-mpnet-base-v2')

def semantic_similarity(expected: str, response: str) -> float:
    """
    Computes the cosine similarity between the expected and response text.
    Returns a float between 0 and 1.
    """
    embedding_expected = similarity_model.encode(expected, convert_to_tensor=True)
    embedding_response = similarity_model.encode(response, convert_to_tensor=True)
    cosine_sim = util.pytorch_cos_sim(embedding_expected, embedding_response)
    return cosine_sim.item()

# Large test dataset based on MedlinePlus diseases.
test_data = [
    {
        "query": "What are the common symptoms of asthma?",
        "expected": "Asthma symptoms include wheezing, shortness of breath, chest tightness, and coughing.",
        "reference": "MedlinePlus. Asthma. https://medlineplus.gov/asthma.html"
    },
    {
        "query": "What is anemia and what causes it?",
        "expected": "Anemia is a condition characterized by a deficiency of red blood cells or hemoglobin, commonly caused by iron deficiency, vitamin deficiencies, or chronic disease.",
        "reference": "MedlinePlus. Anemia. https://medlineplus.gov/anemia.html"
    },
    {
        "query": "What are the early signs of Alzheimer's disease?",
        "expected": "Early signs of Alzheimer's include memory loss, confusion, difficulty performing familiar tasks, and changes in mood or personality.",
        "reference": "MedlinePlus. Alzheimer's Disease. https://medlineplus.gov/alzheimersdisease.html"
    },
    {
        "query": "What types of arthritis exist and how are they treated?",
        "expected": "Common types of arthritis include osteoarthritis and rheumatoid arthritis, treated with pain relievers, anti-inflammatory drugs, physical therapy, and sometimes disease-modifying medications.",
        "reference": "MedlinePlus. Arthritis. https://medlineplus.gov/arthritis.html"
    },
    {
        "query": "What is atrial fibrillation and what are its risks?",
        "expected": "Atrial fibrillation is an irregular and often rapid heart rate that increases the risk of stroke, heart failure, and other heart-related complications.",
        "reference": "MedlinePlus. Atrial Fibrillation. https://medlineplus.gov/atrialfibrillation.html"
    },
   
]

def get_chatbot_response(query: str, context: str = "") -> str:
    """
    Runs your QA chain for a given query.
    """
    input_data = {"query": query, "context": context}
    llm = initialize_llm()
    response = long_running_task(input_data, llm)
    return response.strip()

def evaluate_retrieval(query: str, expected: str, k: int = 5) -> float:
    """
    Evaluates retrieval quality by fetching the top k documents for a query
    and computing the average semantic similarity between each document's content
    and the expected answer.
    """
    retriever = docsearch.as_retriever(search_kwargs={'k': k})
    docs = retriever.invoke(query)
    if not docs:
        return 0.0
    scores = [semantic_similarity(expected, doc.page_content) for doc in docs]
    return np.mean(scores)

def evaluate_responses():
    """
    Evaluates chatbot responses against expected answers using semantic similarity,
    and measures retrieval quality.
    Returns lists of similarity scores and retrieval scores.
    """
    similarity_scores = []
    retrieval_scores = []
    
    for test in test_data:
        query = test["query"]
        expected = test["expected"]
        
        # Retrieve chatbot response.
        response = get_chatbot_response(query, context="")
        sim_score = semantic_similarity(expected, response)
        similarity_scores.append(sim_score)
        
        # Evaluate retrieval quality using top-5 documents.
        ret_score = evaluate_retrieval(query, expected, k=5)
        retrieval_scores.append(ret_score)
        
        print(f"Query:                     {query}")
        print(f"Expected Answer:           {expected}")
        print(f"Chatbot Response:          {response}")
        print(f"Semantic Similarity Score (Generation): {sim_score:.2f}")
        print(f"Average Retrieval Similarity Score (top-5): {ret_score:.2f}")
        print("-" * 80)
    
    return similarity_scores, retrieval_scores

def plot_evaluation(similarity_scores, retrieval_scores):
    """
    Uses Plotly to plot the semantic similarity scores for generation and retrieval,
    and saves the graph as a PNG file.
    """
    test_case_indices = list(range(len(test_data)))
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=test_case_indices,
        y=similarity_scores,
        mode='lines+markers',
        name='Generation Similarity'
    ))
    fig.add_trace(go.Scatter(
        x=test_case_indices,
        y=retrieval_scores,
        mode='lines+markers',
        name='Retrieval Similarity'
    ))
    fig.update_layout(
        title="Semantic Similarity Scores per Test Case",
        xaxis_title="Test Case Index",
        yaxis_title="Similarity Score",
        template="plotly_white"
    )
    fig.show()
    # Save the graph as a PNG file
    fig.write_image("evaluation_graph.png")

if __name__ == "__main__":
    # Evaluate responses and collect scores.
    sim_scores, ret_scores = evaluate_responses()
    
    # Plot evaluation metrics using Plotly.
    plot_evaluation(sim_scores, ret_scores)

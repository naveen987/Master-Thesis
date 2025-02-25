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
    {
        "query": "What are the symptoms and treatment for appendicitis?",
        "expected": "Appendicitis symptoms include sudden abdominal pain, fever, and nausea; treatment typically involves surgical removal of the appendix.",
        "reference": "MedlinePlus. Appendicitis. https://medlineplus.gov/appendicitis.html"
    },
    {
        "query": "What causes acne and how is it treated?",
        "expected": "Acne is caused by clogged hair follicles, excess oil, and bacteria; treatment options include topical medications, antibiotics, and in severe cases, isotretinoin.",
        "reference": "MedlinePlus. Acne. https://medlineplus.gov/acne.html"
    },
    {
        "query": "What is atherosclerosis and what complications can it lead to?",
        "expected": "Atherosclerosis is the buildup of fats and cholesterol in artery walls, which can lead to restricted blood flow, heart attack, or stroke.",
        "reference": "MedlinePlus. Atherosclerosis. https://medlineplus.gov/atherosclerosis.html"
    },
    {
        "query": "What are common symptoms of anxiety disorders?",
        "expected": "Anxiety disorders are characterized by excessive worry, restlessness, increased heart rate, and physical symptoms like sweating.",
        "reference": "MedlinePlus. Anxiety. https://medlineplus.gov/anxiety.html"
    },
    {
        "query": "What are autoimmune diseases and how do they affect the body?",
        "expected": "Autoimmune diseases occur when the immune system mistakenly attacks the body, leading to conditions such as lupus, rheumatoid arthritis, and multiple sclerosis.",
        "reference": "MedlinePlus. Autoimmune Diseases. https://medlineplus.gov/autoimmunediseases.html"
    },
    {
        "query": "What is alcohol-related liver disease and what are its symptoms?",
        "expected": "Alcohol-related liver disease is liver damage caused by excessive alcohol consumption, with symptoms including jaundice, abdominal pain, and fatigue.",
        "reference": "MedlinePlus. Alcohol-Related Liver Disease. https://medlineplus.gov/alcoholrelatedliverdisease.html"
    },
    {
        "query": "What are the dietary recommendations for managing hypertension?",
        "expected": "A low-sodium, balanced diet rich in fruits, vegetables, and whole grains is recommended to help manage hypertension.",
        "reference": "MedlinePlus. High Blood Pressure. https://medlineplus.gov/highbloodpressure.html"
    },
    {
        "query": "How does regular exercise benefit cardiovascular health?",
        "expected": "Regular exercise strengthens the heart, lowers blood pressure, and improves circulation, thereby reducing the risk of heart disease.",
        "reference": "MedlinePlus. Heart Health. https://medlineplus.gov/hearthealth.html"
    },
    {
        "query": "What is the difference between systolic and diastolic blood pressure?",
        "expected": "Systolic blood pressure is the pressure in the arteries during heart contraction, while diastolic is the pressure when the heart rests between beats.",
        "reference": "MedlinePlus. High Blood Pressure. https://medlineplus.gov/highbloodpressure.html"
    },
    {
        "query": "What are the recommended treatments for atrial fibrillation?",
        "expected": "Treatments for atrial fibrillation include medications to control heart rate and rhythm, blood thinners, and in some cases, procedures like electrical cardioversion or ablation.",
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

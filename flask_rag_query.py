import os
import numpy as np
import faiss
import pickle
import torch
from transformers import AutoTokenizer, AutoModel
import google.generativeai as genai
import math
import time
from flask import Flask, render_template, request

app = Flask(__name__)

# --- Configuration ---
FAISS_INDEX_FILE = 'vector_store_hf.index'
CHUNK_DATA_FILE = 'chunks_hf.pkl'
EMBEDDING_MODEL_NAME = 'nomic-ai/nomic-embed-text-v1'
MODEL_MAX_LENGTH = 8192 # Nomic's context length (for embedding)
TOP_K = 5          # Number of chunks to retrieve
GEMINI_MODEL_NAME = 'gemini-1.5-flash-latest' # Or 'gemini-1.0-pro', etc.
API_KEY_FILENAME = 'gemini_api_key.txt' # <<< Name of the file containing the key

# --- Global Variables to Hold Loaded Resources ---
index = None
all_chunks = None
metadata = None
tokenizer = None
model = None
device = None
gemini_model = None
resources_loaded = False
resource_load_error = None

# --- Helper Function for Mean Pooling ---
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask

# --- Load Resources ---
def load_resources():
    global index, all_chunks, metadata, tokenizer, model, device, gemini_model, resources_loaded, resource_load_error
    print("Loading resources...")
    resource_load_error = None
    resources_loaded = False
    # Check for GPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    # Load FAISS index
    try:
        index = faiss.read_index(FAISS_INDEX_FILE)
        print(f"FAISS index loaded. Contains {index.ntotal} vectors.")
    except Exception as e:
        error_msg = f"Error loading FAISS index from {FAISS_INDEX_FILE}: {e}"
        print(error_msg)
        resource_load_error = error_msg
        return

    # Load chunks and metadata
    try:
        with open(CHUNK_DATA_FILE, 'rb') as f:
            chunk_data = pickle.load(f)
        all_chunks = chunk_data['chunks']
        metadata = chunk_data.get('metadata', [])
        print(f"Chunk data loaded. Contains {len(all_chunks)} chunks.")
        if not metadata:
            print("Warning: Metadata not found in chunk data file.")
    except FileNotFoundError:
        error_msg = f"Error: Chunk data file not found at {CHUNK_DATA_FILE}"
        print(error_msg)
        resource_load_error = error_msg
        return
    except Exception as e:
        error_msg = f"Error loading chunk data from {CHUNK_DATA_FILE}: {e}"
        print(error_msg)
        resource_load_error = error_msg
        return

    # Load embedding model
    try:
        print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
        tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME, trust_remote_code=True)
        model = AutoModel.from_pretrained(EMBEDDING_MODEL_NAME, trust_remote_code=True)
        model.to(device)
        model.eval()
        print("Embedding model loaded.")
    except Exception as e:
        error_msg = f"Error loading embedding model {EMBEDDING_MODEL_NAME}: {e}. Ensure 'transformers>=4.36' is installed and you have internet connection."
        print(error_msg)
        resource_load_error = error_msg
        return

    # Configure Gemini API
    try:
        api_key = None
        try:
            with open(API_KEY_FILENAME, 'r') as f:
                api_key = f.readline().strip()

            if not api_key:
                error_msg = f"Error: API key file '{API_KEY_FILENAME}' is empty."
                print(error_msg)
                resource_load_error = error_msg
                return

        except FileNotFoundError:
            error_msg = f"Error: API key file not found at '{API_KEY_FILENAME}'. Please create this file and paste your Gemini API key into it."
            print(error_msg)
            resource_load_error = error_msg
            return
        except Exception as e:
            error_msg = f"Error reading API key file '{API_KEY_FILENAME}': {e}"
            print(error_msg)
            resource_load_error = error_msg
            return

        genai.configure(api_key=api_key)
        generation_config = genai.types.GenerationConfig()
        gemini_model = genai.GenerativeModel(
            GEMINI_MODEL_NAME,
            generation_config=generation_config
        )
        print(f"Gemini model '{GEMINI_MODEL_NAME}' configured.")

    except Exception as e:
        error_msg = f"Error configuring Gemini API: {e}"
        print(error_msg)
        resource_load_error = error_msg
        return

    print("Resources loaded successfully.")
    resources_loaded = True

# --- Embed Query ---
def embed_query(query_text):
    """Embeds a single user query using the loaded Nomic model."""
    prefixed_query = "search_query: " + query_text
    encoded_input = tokenizer(prefixed_query, padding=False, truncation=True, max_length=MODEL_MAX_LENGTH, return_tensors='pt')
    encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
    with torch.no_grad():
        model_output = model(**encoded_input)
    query_embedding = mean_pooling(model_output, encoded_input['attention_mask'])
    query_embedding = torch.nn.functional.normalize(query_embedding, p=2, dim=1)
    return query_embedding.cpu().numpy().astype(np.float32)

# --- Retrieve Chunks ---
def retrieve_chunks(query_embedding):
    """Searches FAISS index and retrieves corresponding text chunks."""
    print(f"Searching for top {TOP_K} relevant chunks...")
    start_time = time.time()
    distances, indices = index.search(query_embedding, TOP_K)
    end_time = time.time()
    print(f"Search completed in {end_time - start_time:.4f} seconds.")
    retrieved_texts = []
    retrieved_metadata = []
    for i, idx in enumerate(indices[0]):
        if idx != -1:
            retrieved_texts.append(all_chunks[idx])
            if metadata:
                retrieved_metadata.append(metadata[idx])
            else:
                retrieved_metadata.append({'index': idx})
        else:
            print(f"  Warning: Retrieved invalid index (-1) at position {i+1}.")
    return retrieved_texts, retrieved_metadata

# --- Generate Response with Gemini ---
def generate_response(query, retrieved_texts):
    """Constructs prompt and calls Gemini API."""
    context = "\n\n---\n\n".join(retrieved_texts)
    prompt = f"""Based ONLY on the following context, please answer the query. If the context does not contain the information needed to answer the query, state that clearly. Do not use any prior knowledge.

Context:
---
{context}
---

Query: {query}

Answer:"""
    print("\n--- Sending Prompt to Gemini ---")
    # print(prompt) # Uncomment to see the full prompt
    print("--- End of Prompt --- \n")
    try:
        print("Waiting for Gemini response...")
        start_time = time.time()
        response = gemini_model.generate_content(prompt)
        end_time = time.time()
        print(f"Gemini response received in {end_time - start_time:.2f} seconds.")

        if response.parts:
            return response.text
        elif response.prompt_feedback.block_reason:
            block_reason = response.prompt_feedback.block_reason
            safety_ratings = response.prompt_feedback.safety_ratings
            details = f"Reason: {block_reason}."
            if safety_ratings:
                details += f" Safety Ratings: {safety_ratings}"
            return f"Blocked by API. {details}"
        else:
            print("Warning: Received empty or unexpected response from Gemini.")
            return "Sorry, I received an unexpected response from the language model."

    except Exception as e:
        print(f"Error calling Gemini API: {e}")
        return "Sorry, I encountered an error while contacting the language model."

# --- Flask Routes ---
@app.route("/", methods=["GET"])
def index():
    if not resources_loaded:
        return render_template("index.html", error=resource_load_error)
    return render_template("index.html")

@app.route("/query", methods=["POST"])
def query():
    if not resources_loaded:
        return render_template("index.html", error=resource_load_error)

    user_query = request.form["query"]
    if not user_query:
        return render_template("index.html", error="Please enter a query.")

    query_embedding = embed_query(user_query)
    retrieved_texts, _ = retrieve_chunks(query_embedding)

    if not retrieved_texts:
        response = "Could not retrieve any relevant context from your documents."
    else:
        response = generate_response(user_query, retrieved_texts)

    return render_template("index.html", response=response, query=user_query)

if __name__ == "__main__":
    load_resources()  # Load resources when the app starts
    if resources_loaded:
        app.run(debug=True)
    else:
        print("Failed to load resources. Flask app will not start.")
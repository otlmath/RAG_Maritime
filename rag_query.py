import os
import numpy as np
import faiss
import pickle
import torch
from transformers import AutoTokenizer, AutoModel
import google.generativeai as genai
import math
import time

# --- Configuration ---
FAISS_INDEX_FILE = 'vector_store_hf.index'
CHUNK_DATA_FILE = 'chunks_hf.pkl'
EMBEDDING_MODEL_NAME = 'nomic-ai/nomic-embed-text-v1'
MODEL_MAX_LENGTH = 8192 # Nomic's context length (for embedding)
TOP_K = 5             # Number of chunks to retrieve
GEMINI_MODEL_NAME = 'gemini-1.5-flash-latest' # Or 'gemini-1.0-pro', etc.
API_KEY_FILENAME = 'gemini_api_key.txt' # <<< Name of the file containing the key

# --- Helper Function for Mean Pooling (same as before) ---
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask

# --- Load Resources ---
def load_resources():
    print("Loading resources...")
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
        print(f"Error loading FAISS index from {FAISS_INDEX_FILE}: {e}")
        return None

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
        print(f"Error: Chunk data file not found at {CHUNK_DATA_FILE}")
        return None
    except Exception as e:
        print(f"Error loading chunk data from {CHUNK_DATA_FILE}: {e}")
        return None

    # Load embedding model
    try:
        print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
        tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME, trust_remote_code=True)
        model = AutoModel.from_pretrained(EMBEDDING_MODEL_NAME, trust_remote_code=True)
        model.to(device)
        model.eval()
        print("Embedding model loaded.")
    except Exception as e:
        print(f"Error loading embedding model {EMBEDDING_MODEL_NAME}: {e}")
        print("Ensure 'transformers>=4.36' is installed and you have internet connection.")
        return None

    # Configure Gemini API - *** MODIFIED SECTION ***
    try:
        api_key = None
        try:
            # Read the API key from the specified file
            with open(API_KEY_FILENAME, 'r') as f:
                api_key = f.readline().strip() # Read the first line and remove whitespace

            if not api_key:
                print(f"Error: API key file '{API_KEY_FILENAME}' is empty.")
                return None
            # print(f"Successfully read API key from {API_KEY_FILENAME}") # Optional debug print

        except FileNotFoundError:
            print(f"Error: API key file not found at '{API_KEY_FILENAME}'.")
            print(f"Please create this file in the same directory as the script,")
            print(f"and paste your Gemini API key into it (and nothing else).")
            return None
        except Exception as e:
             print(f"Error reading API key file '{API_KEY_FILENAME}': {e}")
             return None

        # Configure the genai client with the key read from the file
        genai.configure(api_key=api_key)

        # Setup the specific Gemini model
        generation_config = genai.types.GenerationConfig(
            # temperature=0.7 # Optional: Adjust creativity
        )
        gemini_model = genai.GenerativeModel(
            GEMINI_MODEL_NAME,
            generation_config=generation_config
        )
        print(f"Gemini model '{GEMINI_MODEL_NAME}' configured.")

    except Exception as e:
        # Catch potential errors during genai.configure or GenerativeModel creation
        print(f"Error configuring Gemini API after reading key: {e}")
        return None
    # *** END OF MODIFIED SECTION ***

    print("Resources loaded successfully.")
    return index, all_chunks, metadata, tokenizer, model, device, gemini_model

# --- Embed Query (same as before) ---
def embed_query(query_text, tokenizer, model, device):
    """Embeds a single user query using the loaded Nomic model."""
    prefixed_query = "search_query: " + query_text
    encoded_input = tokenizer(prefixed_query, padding=False, truncation=True, max_length=MODEL_MAX_LENGTH, return_tensors='pt')
    encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
    with torch.no_grad():
        model_output = model(**encoded_input)
    query_embedding = mean_pooling(model_output, encoded_input['attention_mask'])
    query_embedding = torch.nn.functional.normalize(query_embedding, p=2, dim=1)
    return query_embedding.cpu().numpy().astype(np.float32)

# --- Retrieve Chunks (same as before) ---
def retrieve_chunks(query_embedding, index, all_chunks, metadata, k):
    """Searches FAISS index and retrieves corresponding text chunks."""
    print(f"Searching for top {k} relevant chunks...")
    start_time = time.time()
    distances, indices = index.search(query_embedding, k)
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

# --- Generate Response with Gemini (same as before) ---
def generate_response(query, retrieved_texts, gemini_model):
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
        # Use stream=True for potentially faster perceived response time
        # response = gemini_model.generate_content(prompt, stream=True)
        # response.resolve() # Wait for streaming to finish if stream=True used
        response = gemini_model.generate_content(prompt) # Non-streaming
        end_time = time.time()
        print(f"Gemini response received in {end_time - start_time:.2f} seconds.")

        if response.parts:
             return response.text
        elif response.prompt_feedback.block_reason:
             # Providing more specific feedback if blocked
             block_reason = response.prompt_feedback.block_reason
             safety_ratings = response.prompt_feedback.safety_ratings
             details = f"Reason: {block_reason}."
             if safety_ratings:
                 details += f" Safety Ratings: {safety_ratings}"
             return f"Blocked by API. {details}"
        else:
             print("Warning: Received empty or unexpected response from Gemini.")
             # print(response) # Print the full response object for debugging
             return "Sorry, I received an unexpected response from the language model."

    except Exception as e:
        print(f"Error calling Gemini API: {e}")
        # Add more specific error handling if needed, e.g., for google.api_core.exceptions.PermissionDenied
        return "Sorry, I encountered an error while contacting the language model."


# --- Main RAG Loop (same as before) ---
def main():
    resources = load_resources()
    if resources is None:
        print("Failed to initialize resources. Exiting.")
        return

    index, all_chunks, metadata, tokenizer, model, device, gemini_model = resources

    print("\n--- RAG System Ready ---")
    print("Enter your query below. Type 'quit' or 'exit' to stop.")

    while True:
        try:
            user_query = input("\nQuery: ")
            if user_query.lower() in ['quit', 'exit']:
                break
            if not user_query:
                continue

            query_embedding = embed_query(user_query, tokenizer, model, device)
            retrieved_texts, retrieved_meta = retrieve_chunks(query_embedding, index, all_chunks, metadata, TOP_K)

            if not retrieved_texts:
                print("Could not retrieve any relevant context from your documents.")
                # Option 1: Tell the user and stop
                print("Skipping generation step as no context was found.")
                continue
                # Option 2: Ask Gemini without context (will likely hallucinate or use general knowledge)
                # print("Proceeding to ask Gemini without retrieved context.")
                # response_text = generate_response(user_query, [], gemini_model) # Pass empty list
            else:
                # Proceed with context
                response_text = generate_response(user_query, retrieved_texts, gemini_model)

            print("\nGemini Response:")
            print("-" * 20)
            print(response_text)
            print("-" * 20)

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"\nAn unexpected error occurred during the main loop: {e}")


if __name__ == "__main__":
    main()
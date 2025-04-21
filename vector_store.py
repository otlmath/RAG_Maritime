import os
import numpy as np
import faiss
import pickle
import torch
from transformers import AutoTokenizer, AutoModel
import math

# --- Configuration ---
TXT_DIRECTORY = './TXT_DATA' 
EMBEDDING_MODEL_NAME = 'nomic-ai/nomic-embed-text-v1'
MODEL_MAX_LENGTH = 8192 # Max sequence length for nomic-embed-text-v1
FAISS_INDEX_FILE = 'vector_store_hf.index'
CHUNK_DATA_FILE = 'chunks_hf.pkl'
SPLIT_DELIMITER = '############\n'
CHUNK_SIZE = 3
EMBEDDING_BATCH_SIZE = 16 

# --- Initialization ---

# Check for GPU availability
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using GPU:", torch.cuda.get_device_name(0))
else:
    device = torch.device("cpu")
    print("Using CPU")

print(f"Loading Hugging Face tokenizer and model: {EMBEDDING_MODEL_NAME}")
try:
    # Trust remote code for Nomic model
    tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME, trust_remote_code=True)
    model = AutoModel.from_pretrained(EMBEDDING_MODEL_NAME, trust_remote_code=True)
    model.to(device) # Move model to GPU if available
    model.eval() # Set model to evaluation mode
    print("Model and tokenizer loaded.")
except Exception as e:
    print(f"Error loading model or tokenizer: {e}")
    print("Ensure you have 'transformers>=4.36' installed and internet connection.")
    exit()

all_chunks = []
chunk_metadata = []

# --- 1. Read, Split, and Chunk Files ---
print(f"Processing .txt files in directory: {TXT_DIRECTORY}")
try:
    for filename in os.listdir(TXT_DIRECTORY):
        if filename.endswith(".txt"):
            file_path = os.path.join(TXT_DIRECTORY, filename)
            print(f"  Processing: {filename}")
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                sections = content.split(SPLIT_DELIMITER)
                sections = [sec.strip() for sec in sections if sec.strip()]

                for i in range(len(sections) - CHUNK_SIZE + 1):
                    chunk_text = "\n\n".join(sections[i : i + CHUNK_SIZE])
                    all_chunks.append(chunk_text)
                    chunk_metadata.append({
                        'filename': filename,
                        'section_indices': list(range(i, i + CHUNK_SIZE))
                    })
            except Exception as e:
                print(f"    Error processing file {filename}: {e}")

except FileNotFoundError:
    print(f"Error: Directory not found: {TXT_DIRECTORY}")
    exit()
except Exception as e:
    print(f"An unexpected error occurred during file processing: {e}")
    exit()

if not all_chunks:
    print("No chunks were created. Please check your text files and the delimiter.")
    exit()

print(f"Total chunks created: {len(all_chunks)}")

# --- Helper Function for Mean Pooling ---
# (Taken from Sentence Transformers documentation / Hugging Face examples)
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask

# --- 2. Generate Embeddings using Hugging Face (in batches) ---
print(f"Generating embeddings using {EMBEDDING_MODEL_NAME} (Batch size: {EMBEDDING_BATCH_SIZE})...")
all_embeddings_list = []
num_chunks = len(all_chunks)
num_batches = math.ceil(num_chunks / EMBEDDING_BATCH_SIZE)

for i in range(num_batches):
    start_index = i * EMBEDDING_BATCH_SIZE
    end_index = min((i + 1) * EMBEDDING_BATCH_SIZE, num_chunks)
    batch_texts = all_chunks[start_index:end_index]

    print(f"  Processing batch {i+1}/{num_batches} (Chunks {start_index+1}-{end_index})")

    # Tokenize batch
    # Nomic requires adding "search_query: " or "search_document: " prefix depending on use case.
    # For storing documents/chunks, use "search_document: ".
    # Reference: https://huggingface.co/nomic-ai/nomic-embed-text-v1
    batch_texts_prefixed = ["search_document: " + text for text in batch_texts]

    encoded_input = tokenizer(batch_texts_prefixed,
                              padding=True,
                              truncation=True,
                              max_length=MODEL_MAX_LENGTH, # Use the model's max length
                              return_tensors='pt')

    # Move tensors to the correct device
    encoded_input = {k: v.to(device) for k, v in encoded_input.items()}

    # Compute token embeddings with no gradient calculation
    with torch.no_grad():
        model_output = model(**encoded_input)

    # Perform pooling
    batch_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

    # Normalize embeddings (recommended for Nomic)
    batch_embeddings = torch.nn.functional.normalize(batch_embeddings, p=2, dim=1)

    # Move embeddings to CPU and convert to numpy
    all_embeddings_list.append(batch_embeddings.cpu().numpy())

# Concatenate all batch embeddings
embeddings = np.concatenate(all_embeddings_list, axis=0)

print(f"Embeddings generated. Shape: {embeddings.shape}")

# Ensure embeddings are float32, as required by Faiss
embeddings = embeddings.astype('float32')

# --- 3. Create and Populate Faiss Index ---
embedding_dimension = embeddings.shape[1]
print(f"Creating Faiss index (IndexFlatL2) with dimension {embedding_dimension}...")

index = faiss.IndexFlatL2(embedding_dimension)
index.add(embeddings)

print(f"Embeddings added to index. Index total entries: {index.ntotal}")

# --- 4. Save the Index and Chunk Data ---
print(f"Saving Faiss index to: {FAISS_INDEX_FILE}")
faiss.write_index(index, FAISS_INDEX_FILE)

print(f"Saving chunk text and metadata to: {CHUNK_DATA_FILE}")
with open(CHUNK_DATA_FILE, 'wb') as f:
    pickle.dump({'chunks': all_chunks, 'metadata': chunk_metadata}, f)

print("--- Process Complete ---")
print(f"Faiss index saved as '{FAISS_INDEX_FILE}'")
print(f"Chunk data saved as '{CHUNK_DATA_FILE}'")

import os
import numpy as np
import faiss
import pickle
from sentence_transformers import SentenceTransformer
import math
import torch

# --- Configuration ---
TXT_DIRECTORY = './TXT_DATA'
EMBEDDING_MODEL_NAME = 'nomic-ai/nomic-embed-text-v1'
FAISS_INDEX_FILE = 'vector_store_st.index'
CHUNK_DATA_FILE = 'chunks_st.pkl'
SPLIT_DELIMITER = '############\n'
CHUNK_SIZE = 3
EMBEDDING_BATCH_SIZE = 32  # SentenceTransformers often handles larger batches well

# --- Initialization ---

# Check for GPU availability (SentenceTransformers uses PyTorch)
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using GPU:", torch.cuda.get_device_name(0))
else:
    device = torch.device("cpu")
    print("Using CPU")

print(f"Loading Sentence Transformer model: {EMBEDDING_MODEL_NAME}")
try:
    model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=str(device), trust_remote_code=True)
    print("Sentence Transformer model loaded.")
except Exception as e:
    print(f"Error loading Sentence Transformer model: {e}")
    print("Ensure you have 'sentence-transformers' installed and internet connection.")
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

# --- 2. Generate Embeddings using Sentence Transformers ---
print(f"Generating embeddings using {EMBEDDING_MODEL_NAME}...")

# Nomic requires adding "search_document: " prefix for document embeddings.
prefixed_chunks = ["search_document: " + text for text in all_chunks]

embeddings = model.encode(prefixed_chunks,
                         batch_size=EMBEDDING_BATCH_SIZE,
                         show_progress_bar=True,
                         normalize_embeddings=True)

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
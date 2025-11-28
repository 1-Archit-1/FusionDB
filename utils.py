import numpy as np
import faiss
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import torch
import psutil, os

def calculate_recall(found_indices, ground_truth_indices):
    # Convert to sets for efficient intersection
    found_set = set(found_indices)
    truth_set = set(ground_truth_indices)
    
    k = len(truth_set)
    if k == 0:
        # If ground truth is empty, recall is 1.0 (100%) if found is also empty
        # or 0.0 if found is not empty. Or just define as 1.0.
        recall = 1.0 if len(found_set) == 0 else 0.0
    else:
        # Calculate the number of relevant items found
        relevant_found = len(found_set.intersection(truth_set))
        recall = relevant_found / k
    
    return recall

#########################
## Helper Functions ##
#########################

def fetch_test_query(data_embed):
    query_id = 123
    query_vec = data_embed[query_id].astype('float32')
    faiss.normalize_L2(query_vec.reshape(1, -1))
    query_vec = query_vec.reshape(-1) # Make 1D


    # query_1 = "NSFW == 'UNLIKELY'"
    # ground_truth_results = search_baseline_postfilter(query_vec, query_1, res, data_embed, k=10)

    query_2 = "original_width > 1024 and original_height > 1024"


    # query_3 = "similarity > 0.3"
    # search_baseline_postfilter(query_vec, query_3, res, data_embed, k=10)

    return query_vec, query_2

def load_embeddings(file_path):
    print(f"\nLoading embeddings from {file_path} using memory-mapping...")
    try:
        # Load the single, large file using mmap_mode='r'.
        # This does NOT load the file into RAM. It just maps it.
        data_embed = np.load(file_path, mmap_mode='r')
    except FileNotFoundError:
        print(f"Error: File not found: {file_path}")
        print("Please run the 'concatenate_embeddings.py' script first.")
        exit()
    return data_embed

def brute_force_search(query_vector, vectors, k=10):
    if vectors.shape[0] == 0:
        return np.array([]), np.array([])
        
    # Calculate dot product (which is cosine similarity for normalized vectors)
    # We want the highest similarity, so we will sort descending.
    similarities = vectors.dot(query_vector.T)
    
    top_k_indices = np.argsort(similarities)[-k:][::-1]

    top_k_distances = 1.0 - similarities[top_k_indices] # Convert similarity to distance
    
    return top_k_distances, top_k_indices


def batched_vector_search(query_vector, data_embed, ids_to_search, k=10, batch_size=100000):
    """
    Performs a brute-force vector search on a subset of vectors in batches.
    
    Args:
        query_vector: The query vector (normalized, 1D array)
        data_embed: Memory-mapped array of all vectors
        ids_to_search: Array/list of indices to search within data_embed
        k: Number of top results to return
        batch_size: Number of vectors to process per batch
        
    Returns:
        tuple: (result_distances, result_indices) - Top-k distances and their original IDs
    """
    num_to_search = len(ids_to_search)
    
    if num_to_search == 0:
        return np.array([]), np.array([])
    
    # If small enough, process in one go (no batching needed)
    if num_to_search <= batch_size:
        vectors = data_embed[ids_to_search].astype('float32')
        faiss.normalize_L2(vectors)
        distances, local_indices = brute_force_search(query_vector, vectors, k=k)
        result_indices = ids_to_search[local_indices]
        return distances, result_indices
    
    # Process in batches for large sets
    all_distances = []
    all_local_indices = []
    
    for batch_start in range(0, num_to_search, batch_size):
        batch_end = min(batch_start + batch_size, num_to_search)
        batch_ids = ids_to_search[batch_start:batch_end]
        
        # Load and normalize only this batch
        batch_vectors = data_embed[batch_ids].astype('float32')
        faiss.normalize_L2(batch_vectors)
        
        # Search within this batch
        batch_distances, batch_local_indices = brute_force_search(
            query_vector, batch_vectors, k=min(k, len(batch_vectors))
        )
        
        # Adjust indices to be relative to the batch start
        batch_local_indices = batch_local_indices + batch_start
        
        all_distances.extend(batch_distances)
        all_local_indices.extend(batch_local_indices)
    
    # Combine results from all batches and get top-k
    all_distances = np.array(all_distances)
    all_local_indices = np.array(all_local_indices)
    
    # Sort by distance and take top k
    sorted_idx = np.argsort(all_distances)[:k]
    result_distances = all_distances[sorted_idx]
    local_indices = all_local_indices[sorted_idx]
    
    result_indices = ids_to_search[local_indices]
    
    return result_distances, result_indices

def convert_image_to_embedding(image_path): 
    model = SentenceTransformer('clip-ViT-B-32')

    #Encode an image:
    img_emb = model.encode(Image.open(image_path))

    return img_emb

def convert_caption_to_embedding(texts):
    model = SentenceTransformer('clip-ViT-B-32')

    text_emb = model.encode(texts)

    return text_emb

def get_memory_mb():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024**2

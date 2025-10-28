import os
import numpy as np
import pandas as pd
import faiss  # We'll use this for normalization and as a timer/helper
import time
from rel_implementations import duckdb_rel
import utils


#########################
## Helper Functions ##
#########################

def brute_force_search(query_vector, vectors, k=10):
    if vectors.shape[0] == 0:
        return np.array([]), np.array([])
        
    # Calculate dot product (which is cosine similarity for normalized vectors)
    # We want the highest similarity, so we will sort descending.
    similarities = vectors.dot(query_vector.T)
    
    top_k_indices = np.argsort(similarities)[-k:][::-1]

    top_k_distances = 1.0 - similarities[top_k_indices] # Convert similarity to distance
    
    return top_k_distances, top_k_indices



def search_baseline_postfilter(query_vector, sql_where_clause, res, data_embed, k=10):
    """
    Runs the full post-filtering baseline:
    1. Filters metadata *directly from disk* using DuckDB.
    2. Retrieves the filtered vectors *from the memory-mapped file*.
    3. Runs a brute-force search on the filtered vectors.
    
    Returns:
        np.array: The array of original top-k result IDs
    """
    print(f"\n--- Running Post-Filtering Query ---")
    print(f"Filter (SQL): \"{sql_where_clause}\"")

    start_time = time.time()
    unfiltered_vectors = data_embed.astype('float32')
    faiss.normalize_L2(unfiltered_vectors)
    retrieve_time = time.time() - start_time
    print(f"1. Vector retrieval (from mmap): Fetched {len(unfiltered_vectors)} vectors in {retrieve_time:.4f}s")

    start_time = time.time()
    distances, local_indices = brute_force_search(query_vector, unfiltered_vectors, k=k)
    search_time = time.time() - start_time
    print(f"2. Brute-force search: Completed in {search_time:.4f}s")

    start_time = time.time()
    try:
        filtered_ids = duckdb_rel.run_query(res, sql_where_clause)
    except Exception as e:
        print(f"Error applying SQL filter: {e}")
        print("Please check your column names and SQL syntax.")
        return None

    if len(filtered_ids) == 0:
        print("Result: No items matched the metadata filter.")
        return np.array([])
    
    original_indices = np.array([])
    for idx in local_indices:
        if idx in filtered_ids:
            np.append(original_indices, idx)

    if len(local_indices) == 0:
        print("Result: No items matched the metadata filter.")
        return np.array([])

    
    filter_time = time.time() - start_time
    print(f"3. Metadata filtering (from disk): Found {len(original_indices)} items in {filter_time:.4f}s")
     

    return original_indices, filter_time, retrieve_time, search_time, distances


##################
## Main function ##
##################

def baseline_postfilter(meta_path=None, embeddings_path=None):
    METADATA_GLOB_PATH = meta_path
    EMBEDDINGS_FILE_PATH = embeddings_path


    res = duckdb_rel.db_implementation(METADATA_GLOB_PATH)

    print(f"\nLoading embeddings from {EMBEDDINGS_FILE_PATH} using memory-mapping...")
    try:
        # Load the single, large file using mmap_mode='r'.
        # This does NOT load the file into RAM. It just maps it.
        data_embed = np.load(EMBEDDINGS_FILE_PATH, mmap_mode='r')
        
    except FileNotFoundError:
        print(f"Error: File not found: {EMBEDDINGS_FILE_PATH}")
        print("Please run the 'concatenate_embeddings.py' script first.")
        exit()
    num_vectors, d = data_embed.shape
    print(f"Embeddings loaded. Found {num_vectors} vectors of dimension {d}.")


    # Critical Sanity Check:
    print(f"Total metadata rows: {res['total_metadata_rows']}, Total embedding vectors: {num_vectors}")
    assert res['total_metadata_rows'] == num_vectors, "MISMATCH: Metadata and embedding counts do not match!"
    print("Data and metadata counts align. Ready to search.")

    query_id = 123
    query_vec = data_embed[query_id].astype('float32')
    faiss.normalize_L2(query_vec.reshape(1, -1))
    query_vec = query_vec.reshape(-1) # Make 1D


    #### Write the Queries to test here ####
    
    # query_1 = "NSFW == 'UNLIKELY'"
    # ground_truth_results = search_baseline_postfilter(query_vec, query_1, res, data_embed, k=10)

    query_2 = "original_width > 1024 and original_height > 1024"
    ground_truth_results, filter_time, retrieve_time, search_time, distances = search_baseline_postfilter(query_vec, query_2, res, data_embed, k=10)


    # query_3 = "similarity > 0.3"
    # search_baseline_postfilter(query_vec, query_3, res, data_embed, k=10)

    #######################################
    
    if ground_truth_results is not None:
        print("\n\n--- 2. Calculating Recall (example) ---")
        # In the future, 'test_results' will come from your Faiss hybrid system.
        # For this example, we just test the baseline against itself (should be 100%).
        test_results = ground_truth_results # Replace this line later
        
        recall = utils.calculate_recall(found_indices=test_results, 
                        ground_truth_indices=ground_truth_results)
    return {
        'recall': recall, 
        'ground_truth': ground_truth_results,
        'distances': distances,
        'filter_time': filter_time,
        'retrieve_time': retrieve_time,
        'search_time': search_time
    }

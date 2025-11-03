import os
import numpy as np
import pandas as pd
import faiss  # We'll use this for normalization and as a timer/helper
import time
from rel_implementations import duckdb_rel
import utils



def search_baseline_postfilter(query_vector, sql_where_clause, res, data_embed, vector_batch_size=100000, k=10):
    """
    Runs the full post-filtering baseline:
    1. Filters metadata *directly from disk* using DuckDB.
    2. Runs a brute-force search on the vectors.
    3. Filters the top-k results based on the metadata filter.
    
    Returns:
        np.array: The array of original top-k result IDs
    """
    print(f"\n--- Running Post-Filtering Query ---")
    print(f"Filter (SQL): \"{sql_where_clause}\"")

    ##########################################
    ###### Vector Retrieval and Search ######
    ##########################################

    start_time = time.time()
    # Process all vectors in batches to avoid loading entire dataset into memory
    batch_size = vector_batch_size  # Adjust based on available memory
    num_vectors = len(data_embed)
    target_k = 100 * k  # We want to retrieve more candidates for post-filtering
    
    # Create array of all indices to search
    all_ids = np.arange(num_vectors)
    
    # Use batched search to process all vectors
    print(f"1. Processing {num_vectors} vectors in batches of {batch_size}...")
    distances, local_indices = utils.batched_vector_search(
        query_vector, data_embed, all_ids, k=target_k, batch_size=batch_size
    )
    retrieve_time = time.time() - start_time

    ###################################################
    ###### Metadata Post-Filtering of Results #########    
    ###################################################


    print(f"   Vector retrieval and search (batched from mmap): Completed in {retrieve_time:.4f}s")
    print(f"Fetched {len(local_indices)} candidate results before post-filtering.")
    start_time = time.time()
    
    # This filter doesnt use any results from the search itself, 
    # it just gets the IDs that match the SQL clause.
    try:
        filtered_ids = duckdb_rel.run_query(res, sql_where_clause)
    except Exception as e:
        print(f"Error applying SQL filter: {e}")
        print("Please check your column names and SQL syntax.")
        return None

    if len(filtered_ids) == 0:
        print("Result: No items matched the metadata filter.")
        return np.array([])
    
    result_indices = []
    result_distances_list = []
    for i, idx in enumerate(local_indices):
        if idx in filtered_ids:
            result_indices.append(idx)
            result_distances_list.append(distances[i])
            if len(result_indices) == k:
                break
    result_indices = np.array(result_indices)
    result_distances = np.array(result_distances_list)

    filter_time = time.time() - start_time

    print(f"2. Metadata post-filtering: Completed in {filter_time:.4f}s")
    print("----------------------------------------------------------------------")
    return{
        'result_indices': result_indices,
        'filter_time': filter_time,
        'retrieve_time': retrieve_time,  # This now includes both retrieval and search time
        'result_distances': result_distances
    }


##################
## Main function ##
##################

# def baseline_postfilter(meta_path=None, embeddings_path=None):
#     METADATA_GLOB_PATH = meta_path
#     EMBEDDINGS_FILE_PATH = embeddings_path


#     res = duckdb_rel.db_implementation(METADATA_GLOB_PATH)

#     print(f"\nLoading embeddings from {EMBEDDINGS_FILE_PATH} using memory-mapping...")
#     try:
#         # Load the single, large file using mmap_mode='r'.
#         # This does NOT load the file into RAM. It just maps it.
#         data_embed = np.load(EMBEDDINGS_FILE_PATH, mmap_mode='r')
        
#     except FileNotFoundError:
#         print(f"Error: File not found: {EMBEDDINGS_FILE_PATH}")
#         print("Please run the 'concatenate_embeddings.py' script first.")
#         exit()
#     num_vectors, d = data_embed.shape
#     print(f"Embeddings loaded. Found {num_vectors} vectors of dimension {d}.")


#     # Critical Sanity Check:
#     print(f"Total metadata rows: {res['total_metadata_rows']}, Total embedding vectors: {num_vectors}")
#     assert res['total_metadata_rows'] == num_vectors, "MISMATCH: Metadata and embedding counts do not match!"
#     print("Data and metadata counts align. Ready to search.")

#     query_id = 123
#     query_vec = data_embed[query_id].astype('float32')
#     faiss.normalize_L2(query_vec.reshape(1, -1))
#     query_vec = query_vec.reshape(-1) # Make 1D

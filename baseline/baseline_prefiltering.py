import os
import numpy as np
import pandas as pd
import faiss  # We'll use this for normalization and as a timer/helper
import time
from rel_implementations import duckdb_rel
import utils



def search_baseline_prefilter(query_vector, sql_where_clause, res, data_embed, meta_method='duck', vector_batch_size=100000, k=10):
    """
    Runs the full pre-filtering baseline:
    1. Filters metadata *directly from disk* using DuckDB.
    2. Retrieves the filtered vectors *from the memory-mapped file*.
    3. Runs a brute-force search on the filtered vectors.
    
    Returns:
        np.array: The array of original top-k result IDs
    """
    print(f"\n--- Running Pre-Filtering Query ---")
    print(f"Filter (SQL): \"{sql_where_clause}\"")
    
    ##################################################
    ###### Metadata Pre-Filtering of Results #########    
    ##################################################

    start_time = time.time()
    if meta_method == 'duck':
        try:
            filtered_ids = duckdb_rel.run_query(res, sql_where_clause)
        except Exception as e:
            print(f"Error applying SQL filter: {e}")
            print("Please check your column names and SQL syntax.")
            return None
            
        filter_time = time.time() - start_time
        print(f"1. Metadata filtering (from disk): Found {len(filtered_ids)} items in {filter_time:.4f}s")
        
        if len(filtered_ids) == 0:
            print("Result: No items matched the metadata filter.")
            return np.array([])
    else:
        ##### Implement indexed metadata filtering ##########
        pass


    ###################################################
    ########## Vector Retrieval and Search ############
    ###################################################
    start_time = time.time()
    
    # Use batched search to handle large filtered sets efficiently
    batch_size = vector_batch_size  # Adjust based on available memory
    num_filtered = len(filtered_ids)

    result_distances, result_indices = utils.batched_vector_search(
        query_vector, data_embed, filtered_ids, k=k, batch_size=batch_size
    )

    retrieve_time = time.time() - start_time

    if num_filtered <= batch_size:
        print(f"2. Vector retrieval and search: Processed {num_filtered} vectors in {retrieve_time:.4f}s")
    else:
        print(f"   Vector retrieval and search (batched): Processed {num_filtered} vectors in {retrieve_time:.4f}s")
    
    search_time = 0  # Already included in retrieve_time

    print("----------------------------------------------------------------------")
    # return critical data
    return {
        'result_indices': result_indices,
        'filter_time': filter_time,
        'retrieve_time': retrieve_time,
        'search_time': search_time,
        'result_distances': result_distances
    }


##################
## Main function ##
##################

# def baseline_prefilter(meta_path=None, embeddings_path=None):
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


#     #### Write the Queries to test here ####
    
#     # query_1 = "NSFW == 'UNLIKELY'"
#     # ground_truth_results = search_baseline_prefilter(query_vec, query_1, res, data_embed, k=10)

#     query_2 = "original_width > 1024 and original_height > 1024"
#     ground_truth_results, filter_time, retrieve_time, search_time, distances = search_baseline_prefilter(query_vec, query_2, res, data_embed, k=10)


#     # query_3 = "similarity > 0.3"
#     # search_baseline_prefilter(query_vec, query_3, res, data_embed, k=10)

#     #######################################
    
#     if ground_truth_results is not None:
#         print("\n\n--- 2. Calculating Recall (example) ---")
#         # In the future, 'test_results' will come from your Faiss hybrid system.
#         # For this example, we just test the baseline against itself (should be 100%).
#         test_results = ground_truth_results # Replace this line later
        
#         recall = utils.calculate_recall(found_indices=test_results, 
#                         ground_truth_indices=ground_truth_results)
#     return {
#         'recall': recall, 
#         'ground_truth': ground_truth_results,
#         'distances': distances,
#         'filter_time': filter_time,
#         'retrieve_time': retrieve_time,
#         'search_time': search_time
#     }

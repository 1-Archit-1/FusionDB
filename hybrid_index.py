import os
import numpy as np
import pandas as pd
import faiss
import time
from rel_implementations import duckdb_rel, index_rel

def search_hybrid(query_vector, sql_where_clause, res, data_embed, faiss_index, 
                  meta_method='duck', nprobe=10, k=10, selectivity_threshold=0.08):
    """
    Strategy:
    1. Estimate filter selectivity (what % of data matches the metadata filter)
    2. If highly selective (< threshold): Use pre-filtering approach
       - Filter metadata first, then search only filtered vectors with FAISS
    3. If not selective (>= threshold): Use post-filtering approach  
       - Search all vectors with FAISS (fast!), then filter results by metadata
    """
    print(f"\n--- Running Hybrid Search ---")
    print(f"Filter (SQL): \"{sql_where_clause}\"")
    print(f"FAISS nprobe: {nprobe}, k: {k}")
    
    faiss_index.nprobe = nprobe
    total_vectors = len(data_embed)
    
    start_time = time.time()
    if meta_method == 'duck':
        try:
            filtered_ids = duckdb_rel.run_query(res, sql_where_clause)
        except Exception as e:
            print(f"Error applying SQL filter: {e}")
            print("Please check your column names and SQL syntax.")
            return None
    elif meta_method == 'index':
        filtered_ids = index_rel.run_query(res, sql_where_clause)
    
    filter_time = time.time() - start_time
    num_filtered = len(filtered_ids)
    selectivity = num_filtered / total_vectors
    
    print(f"Metadata filter selectivity: {selectivity:.2%} ({num_filtered}/{total_vectors} vectors)")
    
    if num_filtered == 0:
        print("Result: No items matched the metadata filter.")
        return {
            'result_indices': np.array([]),
            'result_distances': np.array([]),
            'filter_time': filter_time,
            'vector_search_time': 0,
            'strategy': 'none'
        }
    
    if selectivity < selectivity_threshold:
        # Pre-filtering: Highly selective filter, search only filtered subset
        strategy = 'pre-filtering'
        print(f"Strategy: PRE-FILTERING (selectivity {selectivity:.2%} < {selectivity_threshold:.2%})")
        
        start_time = time.time()
        
        # Direct search on filtered vectors using FAISS indexFlatIP
        filtered_vectors = data_embed[filtered_ids].astype('float32')
        faiss.normalize_L2(filtered_vectors)
        temp_index = faiss.IndexFlatIP(filtered_vectors.shape[1])  # Inner Product for cosine
        temp_index.add(filtered_vectors)
        
        # Search the filtered subset
        query_reshaped = query_vector.reshape(1, -1).astype('float32')
        similarities, local_indices = temp_index.search(query_reshaped, min(k, num_filtered))
        
        # Convert similarities to distances
        distances = 1.0 - similarities[0]
        local_indices = local_indices[0]
        
        # Map back to original IDs
        result_indices = filtered_ids[local_indices]
        result_distances = distances
        
        vector_search_time = time.time() - start_time
        print(f"Pre-filtering search: {vector_search_time:.4f}s on {num_filtered} vectors")
        
    else:
        # Post-filtering: Less selective, search all with FAISS then filter
        strategy = 'post-filtering'
        print(f"Strategy: POST-FILTERING (selectivity {selectivity:.2%} >= {selectivity_threshold:.2%})")
        
        start_time = time.time()
        
        # Search entire index with FAISS
        target_k = min(100 * k, total_vectors)
        
        query_reshaped = query_vector.reshape(1, -1).astype('float32')
        similarities, candidate_indices = faiss_index.search(query_reshaped, target_k)
        similarities = similarities[0]
        candidate_indices = candidate_indices[0]
        distances = 1.0 - similarities
        
        vector_search_time = time.time() - start_time
        print(f"FAISS search: {vector_search_time:.4f}s, retrieved {len(candidate_indices)} candidates")
        
        start_time = time.time()
        filtered_id_set = set(filtered_ids)
        
        result_indices = []
        result_distances_list = []
        for i, idx in enumerate(candidate_indices):
            if idx in filtered_id_set:
                result_indices.append(idx)
                result_distances_list.append(distances[i])
                if len(result_indices) == k:
                    break
        
        result_indices = np.array(result_indices)
        result_distances = np.array(result_distances_list)
        
        postfilter_time = time.time() - start_time
        filter_time += postfilter_time  # Add post-filtering time to filter_time
        
        print(f"Post-filtering: {postfilter_time:.4f}s, found {len(result_indices)} results")
    
    print(f"Total time: {filter_time + vector_search_time:.4f}s")
    
    return {
        'result_indices': result_indices,
        'result_distances': result_distances,
        'filter_time': filter_time,
        'vector_search_time': vector_search_time,
        'strategy': strategy,
        'selectivity': selectivity
    }


def build_or_load_faiss_index(embeddings_path, index_path=None, index_type='IVFPQ', 
                               nlist=None, m=8, bits=8, rebuild=False):
    """
    Build a new FAISS index or load an existing one.
    """
    # Auto-generate index path
    if index_path is None:
        base_name = os.path.splitext(embeddings_path)[0]
        index_path = f"{base_name}_faiss_{index_type.lower()}.index"
    
    # Try to load existing index
    if os.path.exists(index_path) and not rebuild:
        print(f"Loading existing FAISS index from {index_path}...")
        index = faiss.read_index(index_path)
        print(f"Index loaded: {index.ntotal} vectors")
        return index
    
    print(f"Building new FAISS index ({index_type})...")
    print(f"Loading embeddings from {embeddings_path}...")
    
    data_mmap = np.load(embeddings_path, mmap_mode='r')
    num_vectors, d = data_mmap.shape
    print(f"Found {num_vectors} vectors of dimension {d}")
    
    # For training, we only need a sample
    train_size = min(num_vectors, 100000)  # Use at most 100k vectors for training
    print(f"Using {train_size} vectors for training...")
    
    # Prepare training data
    train_indices = np.random.choice(num_vectors, train_size, replace=False)
    train_data = data_mmap[train_indices].astype('float32')
    faiss.normalize_L2(train_data)
    print(f"Training data prepared and normalized")

    if index_type == 'Flat':
        # Simple flat index
        index = faiss.IndexFlatIP(d)  # Inner Product for cosine similarity
        print("Created IndexFlatIP (exact search, no training needed)")
        
    elif index_type == 'IVF':
        # IVF without product quantization
        if nlist is None:
            nlist = int(np.sqrt(num_vectors))
        quantizer = faiss.IndexFlatIP(d)
        index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
        print(f"Created IndexIVFFlat with {nlist} clusters")
        print("Training index...")
        index.train(train_data)
        print("Training complete")
        
    elif index_type == 'IVFPQ':
        # IVF with product quantization (fast + memory efficient)
        if nlist is None:
            nlist = int(np.sqrt(num_vectors))
        
        # Find valid m
        if d % m != 0:
            for val in [16, 12, 8, 4]:
                if d % val == 0:
                    m = val
                    break
        quantizer = faiss.IndexFlatIP(d)
        index = faiss.IndexIVFPQ(quantizer, d, nlist, m, bits, faiss.METRIC_INNER_PRODUCT)
        print(f"Created IndexIVFPQ with {nlist} clusters, {m} sub-quantizers, {bits} bits")
        print("Training index...")
        index.train(train_data)
        print("Training complete")
    else:
        raise ValueError(f"Unknown index_type: {index_type}")
    
    print("Adding vectors to index in batches...")
    batch_size = 50000
    for i in range(0, num_vectors, batch_size):
        end_idx = min(i + batch_size, num_vectors)
        batch = data_mmap[i:end_idx].astype('float32')
        faiss.normalize_L2(batch)
        index.add(batch)
        print(f"  Added batch {i//batch_size + 1}: {end_idx}/{num_vectors} vectors")
    
    print(f"Added {index.ntotal} vectors total")
    
    if not os.path.exists(os.path.dirname(index_path)) and os.path.dirname(index_path) != '':
        os.makedirs(os.path.dirname(index_path))
    print(f"Saving index to {index_path}...")
    faiss.write_index(index, index_path)
    print("Index saved successfully")
    return index

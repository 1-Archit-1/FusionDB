from baseline.baseline_post_index import search_baseline_postfilter
from baseline.baseline_pre_index import search_baseline_prefilter
from hybrid import search_hybrid, build_or_load_faiss_index
from dotenv import load_dotenv
import os

from rel_implementations import duckdb_rel, index_rel
import faiss
import utils 
import numpy as np

# Load .env
config = load_dotenv()
meta_path = os.getenv('meta_path')
embeddings_path = os.getenv('embeddings_path')

META_METHOD = os.getenv('META_METHOD', 'duck')  # 'duck' or 'index'
print(f"Using metadata method: {META_METHOD}")

METADATA_GLOB_PATH = meta_path
EMBEDDINGS_FILE_PATH = embeddings_path

if META_METHOD == 'duck':
    print("Initializing DuckDB metadata system...")
    res = duckdb_rel.db_implementation(METADATA_GLOB_PATH)
elif META_METHOD == 'index':
    print("Initializing IndexManager metadata system...")
    res = index_rel.db_implementation(METADATA_GLOB_PATH)
else:
    raise ValueError(f"Unknown META_METHOD: {META_METHOD}")

# Load embeddings
data_embed = utils.load_embeddings(EMBEDDINGS_FILE_PATH)
num_vectors, d = data_embed.shape

print(f"Embeddings loaded. Found {num_vectors} vectors of dimension {d}.")

# Critical Sanity Check
print(f"Total metadata rows: {res['total_metadata_rows']}, Total embedding vectors: {num_vectors}")
assert res['total_metadata_rows'] == num_vectors, "MISMATCH: Metadata and embedding counts do not match!"
print("Data and metadata counts align. Ready to search.")

# Build or load FAISS index
print("=======Setting up FAISS index========")
faiss_index = build_or_load_faiss_index(
    embeddings_path=EMBEDDINGS_FILE_PATH,
    index_type='IVFPQ',
    rebuild=False, 
    index_path='faiss_indexes/faiss_index.ivfpq' 
)

query_vec, meta_query = utils.fetch_test_query(data_embed) 
vector_batch_size = int(os.getenv('vector_batch_size'))

print("===== Running Baseline: Pre-filtering =====")
prefilter = search_baseline_prefilter(
    query_vector=query_vec, 
    sql_where_clause=meta_query, 
    res=res, 
    data_embed=data_embed, 
    meta_method=META_METHOD,
    vector_batch_size=vector_batch_size, 
    k=10
)

print("===== Running Baseline: Post-filtering =====")
postfilter = search_baseline_postfilter(
    query_vector=query_vec, 
    sql_where_clause=meta_query, 
    res=res, 
    data_embed=data_embed, 
    meta_method=META_METHOD,
    vector_batch_size=vector_batch_size, 
    k=10
)

print("===== Running Hybrid Search =====")
hybrid = search_hybrid(
    query_vector=query_vec,
    sql_where_clause=meta_query,
    res=res,
    data_embed=data_embed,
    faiss_index=faiss_index,
    meta_method=META_METHOD,
    nprobe=10,
    k=10,
    selectivity_threshold=0.1
)

# Extract results
prefilter_results = prefilter['result_indices']
postfilter_results = postfilter['result_indices']
hybrid_results = hybrid['result_indices']

# Calculate recalls
recall_post_vs_pre = utils.calculate_recall(found_indices=postfilter_results, ground_truth_indices=prefilter_results)
recall_hybrid_vs_pre = utils.calculate_recall(found_indices=hybrid_results, ground_truth_indices=prefilter_results)

print("===== RESULTS COMPARISON =====")
print(f"\nMetadata Method: {META_METHOD}")

print(f"\nPrefilter Indices:  {prefilter_results[:5]}{'...' if len(prefilter_results) > 5 else ''}")
print(f"Postfilter Indices: {postfilter_results[:5]}{'...' if len(postfilter_results) > 5 else ''}")
print(f"Hybrid Indices:     {hybrid_results[:5]}{'...' if len(hybrid_results) > 5 else ''}")

print(f"\nPrefilter distances:  {prefilter['result_distances'][:5]}")
print(f"Postfilter distances: {postfilter['result_distances'][:5]}")
print(f"Hybrid distances:     {hybrid['result_distances'][:5]}")

print(f"\n--- Recall (using Prefilter as ground truth) ---")
print(f"Postfilter recall: {recall_post_vs_pre:.4f}")
print(f"Hybrid recall:     {recall_hybrid_vs_pre:.4f}")

print(f"\n--- Timing Comparison ---")
prefilter_total = prefilter['filter_time'] + prefilter['retrieve_time']
postfilter_total = postfilter['filter_time'] + postfilter['retrieve_time']
hybrid_total = hybrid['filter_time'] + hybrid['vector_search_time']

print(f"Prefilter:  Filter={prefilter['filter_time']:.4f}s, Search={prefilter['retrieve_time']:.4f}s, Total={prefilter_total:.4f}s")
print(f"Postfilter: Search={postfilter['retrieve_time']:.4f}s, Filter={postfilter['filter_time']:.4f}s, Total={postfilter_total:.4f}s")
print(f"Hybrid:     Filter={hybrid['filter_time']:.4f}s, Search={hybrid['vector_search_time']:.4f}s, Total={hybrid_total:.4f}s")
print(f"            Strategy: {hybrid['strategy']}, Selectivity: {hybrid['selectivity']:.2%}")

print(f"\n--- Speedup vs Baselines ---")
if prefilter_total > 0:
    speedup_vs_pre = prefilter_total / hybrid_total
    print(f"Hybrid is {speedup_vs_pre:.2f}x faster than Prefilter")
if postfilter_total > 0:
    speedup_vs_post = postfilter_total / hybrid_total
    print(f"Hybrid is {speedup_vs_post:.2f}x faster than Postfilter")

if META_METHOD == 'index':
    print(f"\n===== IndexManager Performance =====")
    print(f"Your custom tree-based indices are being used for metadata filtering")
    print(f"This demonstrates the benefits of specialized index structures over general-purpose SQL")
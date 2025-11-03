#load environment variables
from baseline.baseline_postfiltering import search_baseline_postfilter
from baseline.baseline_prefiltering import search_baseline_prefilter
from dotenv import load_dotenv
import os
from rel_implementations import duckdb_rel
import faiss
import utils 
import numpy as np


# load .env
config = load_dotenv()
meta_path = os.getenv('meta_path')
embeddings_path = os.getenv('embeddings_path')
METADATA_GLOB_PATH = meta_path
EMBEDDINGS_FILE_PATH = embeddings_path


res = duckdb_rel.db_implementation(METADATA_GLOB_PATH) ### Change here for indexed metadata implementation
data_embed = utils.load_embeddings(EMBEDDINGS_FILE_PATH)
num_vectors, d = data_embed.shape

print(f"Embeddings loaded. Found {num_vectors} vectors of dimension {d}.")
# Critical Sanity Check:
print(f"Total metadata rows: {res['total_metadata_rows']}, Total embedding vectors: {num_vectors}")
assert res['total_metadata_rows'] == num_vectors, "MISMATCH: Metadata and embedding counts do not match!"
print("Data and metadata counts align. Ready to search.")


query_vec, meta_query = utils.fetch_test_query(data_embed) 
vector_batch_size = int(os.getenv('vector_batch_size'))

prefilter = search_baseline_prefilter(
    query_vector = query_vec, 
    sql_where_clause = meta_query, 
    res = res, 
    data_embed = data_embed, 
    meta_method = 'duck', # use duckdb for metadata filtering
    vector_batch_size= vector_batch_size, 
    k=10
)
postfilter = search_baseline_postfilter(
    query_vector = query_vec, 
    sql_where_clause = meta_query, 
    res = res, 
    data_embed = data_embed, 
    meta_method = 'duck', # use duckdb for metadata filtering
    vector_batch_size = vector_batch_size, 
    k = 10
)

prefilter_results = prefilter['result_indices']
postfilter_results = postfilter['result_indices']

recall = utils.calculate_recall(found_indices = postfilter_results, ground_truth_indices=prefilter_results)


print(f"Prefilter Indices: {prefilter_results}")
print(f"Postfilter Indices: {postfilter_results}")
print(f"Prefilter disances: {prefilter['result_distances']}")
print(f"Postfilter distances: {postfilter['result_distances']}")
print(f"Recall of Postfiltering w.r.t Prefiltering: {recall:.4f}")
print(f"Prefilter Times: Filter: {prefilter['filter_time']:.4f}s, Retrieve: {prefilter['retrieve_time']:.4f}s")
print(f"Prefilter total time: {prefilter['filter_time'] + prefilter['retrieve_time']:.4f}s")
print(f"Postfilter Times: Retrieve: {postfilter['retrieve_time']:.4f}s, Filter: {postfilter['filter_time']:.4f}s")
print(f"Postfilter total time: {postfilter['filter_time'] + postfilter['retrieve_time']:.4f}s")
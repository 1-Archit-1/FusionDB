#load environment variables
from dotenv import load_dotenv
import os
from baseline import baseline_prefilter
from baseline import baseline_postfilter

# load .env
config = load_dotenv()
meta_path = os.getenv('meta_path')
embeddings_path = os.getenv('embeddings_path')

#prefilter = baseline_prefilter(meta_path=meta_path, embeddings_path=embeddings_path)
prefilter = baseline_postfilter(meta_path=meta_path, embeddings_path=embeddings_path)
recall = prefilter['recall']
ground_truth = prefilter['ground_truth']
distances = prefilter['distances']
filter_time = prefilter['filter_time']
retrieve_time = prefilter['retrieve_time']
search_time = prefilter['search_time']

print(f"\n--- Baseline Prefiltering Results ---")
print(f"Recall: {recall:.4f}")
print(f"Ground Truth Indices: {ground_truth}")
print(f"Distances: {distances}")
print(f"Filter Time: {filter_time:.4f} seconds")
print(f"Retrieve Time: {retrieve_time:.4f} seconds")
print(f"Search Time: {search_time:.4f} seconds")

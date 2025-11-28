from baseline.baseline_postfiltering import search_baseline_postfilter
from baseline.baseline_prefiltering import search_baseline_prefilter
from dotenv import load_dotenv
import os
from rel_implementations import duckdb_rel
import faiss
import utils 
import numpy as np
import matplotlib.pyplot as plt

# Format
# [0] -> Images
# [1] -> Text
# [2] -> Relational Query
class Evaluator:
    def __init__(self):
        self.workloads = [
            [
                ["Images/orange_cat.webp"],
                [], 
                "nsfw = \'UNSURE\'"
            ],
            [
                ["Images/rat.webp", "Images/man_eating_sandwich.jpg"],
                [],
                "original_width > original_height and license is NULL"
            ],
            [
                [],
                ["people playing poker"],
                "original_width = original_height and license is not NULL and nsfw=\"NSFW\""
            ]
        ]

        self.config = load_dotenv()
        meta_path = os.getenv('meta_path')
        embeddings_path = os.getenv('embeddings_path')
        METADATA_GLOB_PATH = meta_path
        EMBEDDINGS_FILE_PATH = embeddings_path

        self.res = duckdb_rel.db_implementation(METADATA_GLOB_PATH) ### Change here for indexed metadata implementation
        self.data_embed = utils.load_embeddings(EMBEDDINGS_FILE_PATH)
        num_vectors, d = self.data_embed.shape

        print(f"Embeddings loaded. Found {num_vectors} vectors of dimension {d}.")
        # Critical Sanity Check:
        print(f"Total metadata rows: {self.res['total_metadata_rows']}, Total embedding vectors: {num_vectors}")
        assert self.res['total_metadata_rows'] == num_vectors, "MISMATCH: Metadata and embedding counts do not match!"
        print("Data and metadata counts align. Ready to search.")
        self.eval_results = []

    def get_workload(self, idx):
        assert(idx >= 0 and idx < len(self.workloads), "cannot get workload from out of bounds idx")
        
        # Image Embedding
        query_img_vecs = []

        for img_path in self.workloads[idx][0]:
            img_emb = utils.convert_image_to_embedding(img_path)

            query_vec = img_emb.astype('float32')
            faiss.normalize_L2(query_vec.reshape(1, -1))
            query_vec = query_vec.reshape(-1) # Make 1D
            query_img_vecs.append(query_vec)

        # Text Embedding
        text_emb = utils.convert_caption_to_embedding(self.workloads[idx][1])
        query_text_vec =  text_emb.astype('float32')
        faiss.normalize_L2(query_text_vec.reshape(1, -1))
        query_text_vec = query_text_vec.reshape(-1) # Make 1D

        return query_img_vecs, query_text_vec, self.workloads[idx][2]

    def evaluate(self):
        for idx in range(len(self.workloads)):
            img_vecs, text_vec, meta_query = self.get_workload(idx)
            vector_batch_size = 100
            if len(img_vecs) > 0:
                query_vec = img_vecs[0]
            else:
                query_vec = text_vec

            prefilter = search_baseline_prefilter(
                query_vector = query_vec, 
                sql_where_clause = meta_query, 
                res = self.res, 
                data_embed = self.data_embed, 
                meta_method = 'duck', # use duckdb for metadata filtering
                vector_batch_size= vector_batch_size, 
                k=10
            )
            postfilter = search_baseline_postfilter(
                query_vector = query_vec, 
                sql_where_clause = meta_query, 
                res = self.res, 
                data_embed = self.data_embed, 
                meta_method = 'duck', # use duckdb for metadata filtering
                vector_batch_size = vector_batch_size, 
                k = 10
            )
            prefilter_results = prefilter['result_indices']
            postfilter_results = postfilter['result_indices']
            recall = utils.calculate_recall(found_indices = postfilter_results, ground_truth_indices=prefilter_results)

            total_prefilter_time = prefilter['filter_time'] + prefilter['retrieve_time']
            total_postfilter_time = postfilter['filter_time'] + postfilter['retrieve_time']

            # Store results for graphing
            self.eval_results.append({
                "workload": idx,
                "recall": recall,
                "prefilter": {
                    "filter": prefilter['filter_time'],
                    "retrieve": prefilter['retrieve_time'],
                    "total": total_prefilter_time
                },
                "postfilter": {
                    "filter": postfilter['filter_time'],
                    "retrieve": postfilter['retrieve_time'],
                    "total": total_postfilter_time
                }
            })

    def plot_results(self):
        if not self.eval_results:
            print("No results stored. Run evaluate() first.")
            return
        
        workloads = [res["workload"] for res in self.eval_results]
        recall_vals = [res["recall"] for res in self.eval_results]

        # Latencies by type
        pre_total = [res["prefilter"]["total"] for res in self.eval_results]
        post_total = [res["postfilter"]["total"] for res in self.eval_results]

        pre_filter_times = [res["prefilter"]["filter"] for res in self.eval_results]
        post_filter_times = [res["postfilter"]["filter"] for res in self.eval_results]

        pre_retrieve_times = [res["prefilter"]["retrieve"] for res in self.eval_results]
        post_retrieve_times = [res["postfilter"]["retrieve"] for res in self.eval_results]

        fig, ax = plt.subplots(1, 3, figsize=(18, 5))

        # Recall@K
        ax[0].plot(workloads, recall_vals, marker='o', label="Recall@10")
        ax[0].set_title("Recall@10 per Workload")
        ax[0].set_xlabel("Workload ID")
        ax[0].set_ylabel("Recall")
        ax[0].set_ylim(0, 1.05)
        ax[0].grid(True)

        # Total latency
        width = 0.35
        x = np.arange(len(workloads))
        ax[1].bar(x - width/2, pre_total, width, label='Prefilter Total')
        ax[1].bar(x + width/2, post_total, width, label='Postfilter Total')
        ax[1].set_title("Total Latency per Workload")
        ax[1].set_xlabel("Workload ID")
        ax[1].set_ylabel("Seconds")
        ax[1].legend()
        ax[1].grid(True)

        # Breakdown filter vs retrieve
        ax[2].bar(x - width/2, pre_filter_times, width, label='Prefilter Filter')
        ax[2].bar(x + width/2, post_filter_times, width, label='Postfilter Filter')
        ax[2].plot(x, pre_retrieve_times, marker='x', linestyle="--", label='Prefilter Retrieve')
        ax[2].plot(x, post_retrieve_times, marker='x', linestyle="--", label='Postfilter Retrieve')
        ax[2].set_title("Filtering vs Retrieval Time Breakdown")
        ax[2].set_xlabel("Workload ID")
        ax[2].set_ylabel("Seconds")
        ax[2].legend()
        ax[2].grid(True)

        plt.tight_layout()
        plt.show()
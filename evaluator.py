from baseline.baseline_post_index import search_baseline_postfilter
from baseline.baseline_pre_index import search_baseline_prefilter
from hybrid_index import search_hybrid, build_or_load_faiss_index
from dotenv import load_dotenv
import os
from rel_implementations import duckdb_rel, index_rel
import faiss
import utils 
import numpy as np
import matplotlib.pyplot as plt

# Format
# [0] -> Images
# [1] -> Text
# [2] -> Relational Query
class Evaluator:
    def __init__(self, meta_method = 'duck'):
        """
        Args:
            meta_method (string): determine the type of metadata you are dealing with
            choose between 'duck' or 'index'
        """
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

        self.META_METHOD = meta_method

        print(f"Using metadata method: {self.META_METHOD}")

        if self.META_METHOD == 'duck':
            print("Initializing DuckDB metadata system...")
            self.res = duckdb_rel.db_implementation(METADATA_GLOB_PATH)
        elif self.META_METHOD == 'index':
            print("Initializing IndexManager metadata system...")
            self.res = index_rel.db_implementation(METADATA_GLOB_PATH)
        else:
            raise ValueError(f"Unknown META_METHOD: {self.META_METHOD}")
        
        self.data_embed = utils.load_embeddings(EMBEDDINGS_FILE_PATH)
        num_vectors, d = self.data_embed.shape

        print(f"Embeddings loaded. Found {num_vectors} vectors of dimension {d}.")

        # Critical Sanity Check:
        print(f"Total metadata rows: {self.res['total_metadata_rows']}, Total embedding vectors: {num_vectors}")
        assert self.res['total_metadata_rows'] == num_vectors, "MISMATCH: Metadata and embedding counts do not match!"
        print("Data and metadata counts align. Ready to search.")

        # Build or load FAISS index
        print("=======Setting up FAISS index========")
        self.faiss_index = build_or_load_faiss_index(
            embeddings_path=EMBEDDINGS_FILE_PATH,
            index_type='IVFPQ',
            rebuild=False, 
            index_path='faiss_indexes/faiss_index.ivfpq' 
        )

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
            vector_batch_size = int(os.getenv('vector_batch_size'))
            if len(img_vecs) > 0:
                query_vec = img_vecs[0]
            else:
                query_vec = text_vec

            print("===== Running Baseline: Pre-filtering =====")
            prefilter = search_baseline_prefilter(
                query_vector = query_vec, 
                sql_where_clause = meta_query, 
                res = self.res, 
                data_embed = self.data_embed, 
                meta_method = self.META_METHOD, # use duckdb for metadata filtering
                vector_batch_size= vector_batch_size, 
                k=10
            )

            print("===== Running Baseline: Post-filtering =====")
            postfilter = search_baseline_postfilter(
                query_vector = query_vec, 
                sql_where_clause = meta_query, 
                res = self.res, 
                data_embed = self.data_embed, 
                meta_method =self.META_METHOD, # use duckdb for metadata filtering
                vector_batch_size = vector_batch_size, 
                k = 10
            )

            print("===== Running Hybrid Search =====")
            hybrid = search_hybrid(
                query_vector=query_vec,
                sql_where_clause=meta_query,
                res=self.res,
                data_embed=self.data_embed,
                faiss_index=self.faiss_index,
                meta_method=self.META_METHOD,
                nprobe=10,
                k=10,
                selectivity_threshold=0.1
            )


            # Indcies
            prefilter_results = prefilter['result_indices']
            postfilter_results = postfilter['result_indices']
            hybrid_results = hybrid['result_indices']

            # Recall
            recall_post_vs_pre = utils.calculate_recall(found_indices=postfilter_results, ground_truth_indices=prefilter_results)
            recall_hybrid_vs_pre = utils.calculate_recall(found_indices=hybrid_results, ground_truth_indices=prefilter_results)

            # Time
            total_prefilter_time = prefilter['filter_time'] + prefilter['retrieve_time']
            total_postfilter_time = postfilter['filter_time'] + postfilter['retrieve_time']
            total_hybrid_time = hybrid['filter_time'] + hybrid['vector_search_time']

            # Memory Usage
            total_prefilter_mem = prefilter['filter_mem'] + prefilter['retrieve_mem']
            total_postfilter_mem = postfilter['filter_mem'] + postfilter['retrieve_mem']
            total_hybrid_mem = hybrid['filter_mem'] + hybrid['vector_search_mem']

            # Store results for graphing
            self.eval_results.append({
                "workload": idx,
                "recall_post_vs_pre": recall_post_vs_pre,
                "recall_hybrid_vs_pre": recall_hybrid_vs_pre,
                "prefilter": {
                    "filter_time": prefilter['filter_time'],
                    "retrieve_time": prefilter['retrieve_time'],
                    "total_time": total_prefilter_time,
                    "filter_mem":prefilter['filter_mem'],
                    "retrieve_mem":prefilter['retrieve_mem'],
                    "total_mem": total_prefilter_mem,
                },
                "postfilter": {
                    "filter_time": postfilter['filter_time'],
                    "retrieve_time": postfilter['retrieve_time'],
                    "total_time": total_postfilter_time,
                    "filter_mem": postfilter['filter_mem'],
                    "retrieve_mem": postfilter['retrieve_mem'],
                    "total_mem": total_postfilter_mem,
                },
                "hybrid":{
                    "filter_time": hybrid['filter_time'],
                    "vec_search_time": hybrid['vector_search_time'],
                    "total_time": total_hybrid_time,
                    "filter_mem": hybrid['filter_mem'],
                    "vec_search_mem": hybrid['vector_search_mem'],
                    "total_mem": total_hybrid_mem,
                }
            })

    def export_results(self, output_dir="results"):
        """
        Saves a text report for each workload based on eval_results.
        Must be called AFTER evaluate().
        """

        if not self.eval_results:
            print("No evaluation results found. Run evaluate() first.")
            return
        
        # Create folder if not exists
        os.makedirs(output_dir, exist_ok=True)

        for result in self.eval_results:
            workload_id = result["workload"]
            filepath = os.path.join(output_dir, f"workload_{workload_id}.txt")

            with open(filepath, "w") as f:
                f.write(f"===== WORKLOAD {workload_id} RESULTS =====\n\n")

                # Recalls
                f.write(f"--- Recall (vs Prefilter Ground Truth) ---\n")
                f.write(f"Postfilter Recall: {result['recall_post_vs_pre']:.4f}\n")
                f.write(f"Hybrid Recall: {result['recall_hybrid_vs_pre']:.4f}\n")

                f.write("\n")

                # Prefilter performance
                prefilter = result["prefilter"]
                f.write(f"--- Prefilter ---\n")
                f.write(f"Filter Time: {prefilter['filter_time']:.4f}s\n")
                f.write(f"Retrieve Time: {prefilter['retrieve_time']:.4f}s\n")
                f.write(f"Total Time: {prefilter['total_time']:.4f}s\n")
                f.write(f"Filter Mem: {prefilter['filter_mem']:.2f}MB\n")
                f.write(f"Retrieval Mem: {prefilter['retrieve_mem']:.2f}MB\n")
                f.write(f"Total Mem: {prefilter['total_mem']:.2f}MB\n")

                f.write("\n")

                # Postfilter performance
                postfilter = result["postfilter"]
                f.write(f"--- Postfilter ---\n")
                f.write(f"Filter Time: {postfilter['filter_time']:.4f}s\n")
                f.write(f"Retrieve Time: {postfilter['retrieve_time']:.4f}s\n")
                f.write(f"Total Time: {postfilter['total_time']:.4f}s\n")
                f.write(f"Filter Mem: {postfilter['filter_mem']:.2f}MB\n")
                f.write(f"Retrieval Mem: {postfilter['retrieve_mem']:.2f}MB\n")
                f.write(f"Total Mem: {postfilter['total_mem']:.2f}MB\n")

                f.write("\n")

                # Hybrid performance
                hybrid = result["hybrid"]
                f.write(f"--- Hybrid Index Search ---\n")
                f.write(f"Filter Time: {hybrid['filter_time']:.4f}s\n")
                f.write(f"Vector Search Time: {hybrid['vec_search_time']:.4f}s\n")
                f.write(f"Total Time: {hybrid['total_time']:.4f}s\n")
                f.write(f"Filter Mem: {hybrid['filter_mem']:.2f}MB\n")
                f.write(f"Vector Search Mem: {hybrid['vec_search_mem']:.2f}MB\n")
                f.write(f"Total Mem: {hybrid['total_mem']:.2f}MB\n")

                f.write("\n")

                f.write(f"--- Time Speedup ---\n")
                if prefilter['total_time'] > 0 and postfilter['total_time'] > 0:
                    pre_vs_post = postfilter['total_time']  / prefilter['total_time']
                    f.write(f"Prefilter is {pre_vs_post:.2f}x faster than Postfilter\n")
                if prefilter['total_time'] > 0:
                    speedup_vs_pre = prefilter['total_time'] / hybrid['total_time']
                    f.write(f"Hybrid is {speedup_vs_pre:.2f}x faster than Prefilter\n")
                if postfilter['total_time'] > 0:
                    speedup_vs_post = postfilter['total_time'] / hybrid['total_time']
                    f.write(f"Hybrid is {speedup_vs_post:.2f}x faster than Postfilter\n")

                f.write("\n")

                f.write(f"--- Memory Consumption Changes ---\n")
                usage_vs_pre = prefilter['total_mem'] / hybrid['total_mem']
                usage_vs_post = postfilter['total_mem'] / hybrid['total_mem']
                f.write(f"Hybrid Uses {usage_vs_pre:.2f}x more memory than Prefilter\n")
                f.write(f"Hybrid Uses {usage_vs_post:.2f}x more memory Postfilter\n")
                    

        print("\nAll workload results exported successfully!")        
    


    def plot_results(self):
        if not self.eval_results:
            print("No results stored. Run evaluate() first.")
            return

        workloads = [res["workload"] for res in self.eval_results]

        # Recall
        recall_post_vs_pre = [res["recall_post_vs_pre"] for res in self.eval_results]
        recall_hybrid_vs_pre = [res["recall_hybrid_vs_pre"] for res in self.eval_results]

        # Latency breakdown
        pre_filter = [res["prefilter"]["filter_time"] for res in self.eval_results]
        pre_retrieve = [res["prefilter"]["retrieve_time"] for res in self.eval_results]
        pre_total = [res["prefilter"]["total_time"] for res in self.eval_results]

        post_filter = [res["postfilter"]["filter_time"] for res in self.eval_results]
        post_retrieve = [res["postfilter"]["retrieve_time"] for res in self.eval_results]
        post_total = [res["postfilter"]["total_time"] for res in self.eval_results]

        hybrid_filter = [res["hybrid"]["filter_time"] for res in self.eval_results]
        hybrid_vec = [res["hybrid"]["vec_search_time"] for res in self.eval_results]
        hybrid_total = [res["hybrid"]["total_time"] for res in self.eval_results]

        # Speed-ups
        speed_post_vs_pre = [pre/post if post > 0 else np.nan for pre, post in zip(pre_total, post_total)]
        speed_hybrid_vs_pre = [pre/hyb if hyb > 0 else np.nan for pre, hyb in zip(pre_total, hybrid_total)]
        speed_hybrid_vs_post = [post/hyb if hyb > 0 else np.nan for post, hyb in zip(post_total, hybrid_total)]

        # Memory usage
        pre_mem = [res["prefilter"]["total_mem"] for res in self.eval_results]
        post_mem = [res["postfilter"]["total_mem"] for res in self.eval_results]
        hybrid_mem = [res["hybrid"]["total_mem"] for res in self.eval_results]

        x = np.arange(len(workloads))
        width = 0.25

        fig, axes = plt.subplots(3, 2, figsize=(18, 15))

        # Recall
        axes[0,0].plot(workloads, recall_post_vs_pre, marker='o', label="Post vs Pre Recall")
        axes[0,0].plot(workloads, recall_hybrid_vs_pre, marker='x', label="Hybrid vs Pre Recall")
        axes[0,0].set_title("Recall per Workload")
        axes[0,0].set_xlabel("Workload ID")
        axes[0,0].set_ylabel("Recall")
        axes[0,0].set_ylim(0, 1.05)
        axes[0,0].grid(True)
        axes[0,0].legend()

        # Total Latency
        axes[0,1].bar(x - width, pre_total, width, label='Prefilter')
        axes[0,1].bar(x, post_total, width, label='Postfilter')
        axes[0,1].bar(x + width, hybrid_total, width, label='Hybrid')
        axes[0,1].set_title("Total Latency per Workload")
        axes[0,1].set_xlabel("Workload ID")
        axes[0,1].set_ylabel("Seconds")
        axes[0,1].legend()
        axes[0,1].grid(True)

        # Filter vs Retrieve Time Breakdown
        axes[1,0].bar(x - width, pre_filter, width, label='Prefilter Filter')
        axes[1,0].bar(x - width, pre_retrieve, width, bottom=pre_filter, label='Prefilter Retrieve')

        axes[1,0].bar(x, post_filter, width, label='Postfilter Filter')
        axes[1,0].bar(x, post_retrieve, width, bottom=post_filter, label='Postfilter Retrieve')

        axes[1,0].bar(x + width, hybrid_filter, width, label='Hybrid Filter')
        axes[1,0].bar(x + width, hybrid_vec, width, bottom=hybrid_filter, label='Hybrid Vec Search')

        axes[1,0].set_title("Filter vs Retrieval Time Breakdown")
        axes[1,0].set_xlabel("Workload ID")
        axes[1,0].set_ylabel("Seconds")
        axes[1,0].legend()
        axes[1,0].grid(True)

        # Speed-ups
        axes[1,1].plot(x, speed_post_vs_pre, marker='o', label="Post vs Pre")
        axes[1,1].plot(x, speed_hybrid_vs_pre, marker='x', label="Hybrid vs Pre")
        axes[1,1].plot(x, speed_hybrid_vs_post, marker='^', label="Hybrid vs Post")
        axes[1,1].axhline(1.0, linestyle='--', color='gray')
        axes[1,1].set_title("Speed-up Factors (Higher = Faster)")
        axes[1,1].set_ylabel("× Faster")
        axes[1,1].set_xlabel("Workload ID")
        axes[1,1].legend()
        axes[1,1].grid(True)

        # Memory Usage
        axes[2,0].bar(x - width, pre_mem, width, label='Prefilter')
        axes[2,0].bar(x, post_mem, width, label='Postfilter')
        axes[2,0].bar(x + width, hybrid_mem, width, label='Hybrid')
        axes[2,0].set_title("Memory Usage per Workload (MB)")
        axes[2,0].set_ylabel("MB")
        axes[2,0].set_xlabel("Workload ID")
        axes[2,0].legend()
        axes[2,0].grid(True)

        # Hide the empty subplot (axes[2,1])
        axes[2,1].axis('off')

        plt.tight_layout()
        plt.show()
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
                "nsfw = \'UNSURE\'",
                "AND"
            ],
            [
                ["Images/rat.webp", "Images/man_eating_sandwich.jpg"],
                [],
                "original_width > original_height and license is NULL",
                "OR"
            ],
            [
                [],
                ["people in starbuckss"],
                "original_width = original_height and license is not NULL and nsfw=\'NSFW\'",
                "AND"
            ],
            [
                [],
                ['North Shore'],
                "license is NULL",
                "AND"
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
        self.num_runs = 5  # Number of times to run each workload

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

        return query_img_vecs, query_text_vec, self.workloads[idx][2], self.workloads[idx][3]
    
    def get_group_query(self, img_vecs, text_vec):
        all_vecs = []

        # Add image vectors
        for vec in img_vecs:
            if vec is not None and len(vec) > 0:
                all_vecs.append(vec)

        # Add text vector (if any)
        if text_vec is not None and len(text_vec) > 0:
            all_vecs.append(text_vec)

        if not all_vecs:
            raise ValueError("No vectors provided for query.")

        # Stack + Normalize + Mean-pool + Normalize again
        queries = np.stack(all_vecs).astype('float32')
        faiss.normalize_L2(queries)

        combined = np.mean(queries, axis=0, keepdims=False).astype('float32')
        faiss.normalize_L2(combined.reshape(1, -1))

        return combined

    def evaluate(self):
        for idx in range(len(self.workloads)):
            print(f"\n{'='*80}")
            print(f"WORKLOAD {idx}: Running {self.num_runs} iterations")
            print(f"{'='*80}")
            
            img_vecs, text_vec, meta_query, multi_query_mode = self.get_workload(idx)
            vector_batch_size = int(os.getenv('vector_batch_size'))
            query_vec = self.get_group_query(img_vecs, text_vec)

            # Store results from all runs
            run_results = {
                'prefilter': [],
                'postfilter': [],
                'hybrid': []
            }

            # Run workload multiple times
            for run in range(self.num_runs):
                print(f"\n--- Run {run + 1}/{self.num_runs} ---")
                
                print("===== Running Baseline: Pre-filtering =====")
                prefilter = search_baseline_prefilter(
                    query_vector = query_vec, 
                    sql_where_clause = meta_query, 
                    res = self.res, 
                    data_embed = self.data_embed, 
                    meta_method = self.META_METHOD,
                    vector_batch_size= vector_batch_size, 
                    k=10
                )

                print("===== Running Baseline: Post-filtering =====")
                postfilter = search_baseline_postfilter(
                    query_vector = query_vec, 
                    sql_where_clause = meta_query, 
                    res = self.res, 
                    data_embed = self.data_embed, 
                    meta_method =self.META_METHOD,
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

                run_results['prefilter'].append(prefilter)
                run_results['postfilter'].append(postfilter)
                run_results['hybrid'].append(hybrid)

            # Calculate averages
            print(f"\n{'='*80}")
            print(f"Calculating averages for workload {idx}")
            print(f"{'='*80}")
            
            # Average prefilter
            avg_prefilter = {
                'filter_time': np.mean([r['filter_time'] for r in run_results['prefilter']]),
                'retrieve_time': np.mean([r['retrieve_time'] for r in run_results['prefilter']]),
                'filter_mem': np.mean([r['filter_mem'] for r in run_results['prefilter']]),
                'retrieve_mem': np.mean([r['retrieve_mem'] for r in run_results['prefilter']]),
            }
            avg_prefilter['total_time'] = avg_prefilter['filter_time'] + avg_prefilter['retrieve_time']
            avg_prefilter['total_mem'] = avg_prefilter['filter_mem'] + avg_prefilter['retrieve_mem']
            
            # Average postfilter
            avg_postfilter = {
                'filter_time': np.mean([r['filter_time'] for r in run_results['postfilter']]),
                'retrieve_time': np.mean([r['retrieve_time'] for r in run_results['postfilter']]),
                'filter_mem': np.mean([r['filter_mem'] for r in run_results['postfilter']]),
                'retrieve_mem': np.mean([r['retrieve_mem'] for r in run_results['postfilter']]),
            }
            avg_postfilter['total_time'] = avg_postfilter['filter_time'] + avg_postfilter['retrieve_time']
            avg_postfilter['total_mem'] = avg_postfilter['filter_mem'] + avg_postfilter['retrieve_mem']
            
            # Average hybrid
            avg_hybrid = {
                'filter_time': np.mean([r['filter_time'] for r in run_results['hybrid']]) ,
                'vec_search_time': np.mean([r['vector_search_time'] for r in run_results['hybrid']]),
                'filter_mem': np.mean([r['filter_mem'] for r in run_results['hybrid']]),
                'vec_search_mem': np.mean([r['vector_search_mem'] for r in run_results['hybrid']]),
            }
            avg_hybrid['total_time'] = avg_hybrid['filter_time'] + avg_hybrid['vec_search_time']
            avg_hybrid['total_mem'] = avg_hybrid['filter_mem'] + avg_hybrid['vec_search_mem']

            # Use results from last run for indices (should be same across runs)
            prefilter_results = run_results['prefilter'][-1]['result_indices']
            postfilter_results = run_results['postfilter'][-1]['result_indices']
            hybrid_results = run_results['hybrid'][-1]['result_indices']

            # Recall
            recall_post_vs_pre = utils.calculate_recall(found_indices=postfilter_results, ground_truth_indices=prefilter_results)
            recall_hybrid_vs_pre = utils.calculate_recall(found_indices=hybrid_results, ground_truth_indices=prefilter_results)

            # Store averaged results
            self.eval_results.append({
                "workload": idx,
                "recall_post_vs_pre": recall_post_vs_pre,
                "recall_hybrid_vs_pre": recall_hybrid_vs_pre,
                "prefilter": avg_prefilter,
                "postfilter": avg_postfilter,
                "hybrid": avg_hybrid
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
                if prefilter['total_mem'] > 0:
                    usage_vs_pre = hybrid['total_mem'] / prefilter['total_mem'] 
                    f.write(f"Hybrid Uses {usage_vs_pre:.2f}x more memory than Prefilter\n")
                else:
                    f.write(f"Hybrid Uses inf_x more memory than Prefilter\n")

                if postfilter['total_mem'] > 0:
                    usage_vs_post = hybrid['total_mem'] / postfilter['total_mem'] 
                    f.write(f"Hybrid Uses {usage_vs_post:.2f}x more memory Postfilter\n")
                else:
                    f.write(f"Hybrid Uses inf_x more memory than Postfilter\n")
                
                
                    

        print("\nAll workload results exported successfully!")


    def export_averaged_table(self, output_file="results/averaged_results.csv"):
        """
        Exports averaged results across all workloads as a CSV table.
        Must be called AFTER evaluate().
        """
        if not self.eval_results:
            print("No evaluation results found. Run evaluate() first.")
            return
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Prepare data for CSV
        rows = []
        headers = [
            "Workload",
            "Strategy",
            "Filter Time (s)",
            "Retrieve/VecSearch Time (s)",
            "Total Time (s)",
            "Filter Mem (MB)",
            "Retrieve/VecSearch Mem (MB)",
            "Total Mem (MB)",
            "Recall vs Prefilter"
        ]
        
        for result in self.eval_results:
            workload_id = result["workload"]
            
            # Prefilter row
            pre = result["prefilter"]
            rows.append([
                workload_id,
                "Prefilter",
                f"{pre['filter_time']:.4f}",
                f"{pre['retrieve_time']:.4f}",
                f"{pre['total_time']:.4f}",
                f"{pre['filter_mem']:.2f}",
                f"{pre['retrieve_mem']:.2f}",
                f"{pre['total_mem']:.2f}",
                "1.0000"  # Ground truth
            ])
            
            # Postfilter row
            post = result["postfilter"]
            rows.append([
                workload_id,
                "Postfilter",
                f"{post['filter_time']:.4f}",
                f"{post['retrieve_time']:.4f}",
                f"{post['total_time']:.4f}",
                f"{post['filter_mem']:.2f}",
                f"{post['retrieve_mem']:.2f}",
                f"{post['total_mem']:.2f}",
                f"{result['recall_post_vs_pre']:.4f}"
            ])
            
            # Hybrid row
            hyb = result["hybrid"]
            rows.append([
                workload_id,
                "Hybrid",
                f"{hyb['filter_time']:.4f}",
                f"{hyb['vec_search_time']:.4f}",
                f"{hyb['total_time']:.4f}",
                f"{hyb['filter_mem']:.2f}",
                f"{hyb['vec_search_mem']:.2f}",
                f"{hyb['total_mem']:.2f}",
                f"{result['recall_hybrid_vs_pre']:.4f}"
            ])
        
        # Write to CSV
        with open(output_file, 'w', newline='') as f:
            import csv
            writer = csv.writer(f)
            writer.writerow(headers)
            writer.writerows(rows)
        
        print(f"\nAveraged results table exported to: {output_file}")
        
        # Also create a formatted text version for easy reading
        text_output = output_file.replace('.csv', '.txt')
        with open(text_output, 'w') as f:
            f.write(f"AVERAGED RESULTS ACROSS {self.num_runs} RUNS\n")
            f.write("="*120 + "\n\n")
            
            # Calculate column widths
            col_widths = [max(len(str(row[i])) for row in [headers] + rows) + 2 for i in range(len(headers))]
            
            # Write header
            header_line = "".join(h.ljust(col_widths[i]) for i, h in enumerate(headers))
            f.write(header_line + "\n")
            f.write("-"*120 + "\n")
            
            # Write rows
            for row in rows:
                row_line = "".join(str(cell).ljust(col_widths[i]) for i, cell in enumerate(row))
                f.write(row_line + "\n")
            
            f.write("\n" + "="*120 + "\n")
            f.write(f"\nNote: Each workload was run {self.num_runs} times and results were averaged.\n")
            f.write("Recall is calculated with Prefilter results as ground truth.\n")
        
        print(f"Formatted text table exported to: {text_output}")
        print(f"\nSummary: Exported results for {len(self.eval_results)} workloads, {len(rows)} total rows")


    def plot_results(self, save_path=None):
        if not self.eval_results:
            print("No results stored. Run evaluate() first.")
            return

        workloads = [res["workload"] for res in self.eval_results]
        workload_labels = [str(w+1) for w in workloads]  # e.g., 1,2,3 or custom labels

        # Recall values
        recall_post = [res["recall_post_vs_pre"] for res in self.eval_results]
        recall_hybrid = [res["recall_hybrid_vs_pre"] for res in self.eval_results]

        # Total latencies
        pre_total = [res["prefilter"]["total_time"] for res in self.eval_results]
        post_total = [res["postfilter"]["total_time"] for res in self.eval_results]
        hybrid_total = [res["hybrid"]["total_time"] for res in self.eval_results]

        # Memory usage
        pre_mem = [res["prefilter"]["total_mem"] for res in self.eval_results]
        post_mem = [res["postfilter"]["total_mem"] for res in self.eval_results]
        hybrid_mem = [res["hybrid"]["total_mem"] for res in self.eval_results]

        # Breakdown filter vs retrieval / vector search times
        pre_filter = [res["prefilter"]["filter_time"] for res in self.eval_results]
        pre_retrieve = [res["prefilter"]["retrieve_time"] for res in self.eval_results]
        post_filter = [res["postfilter"]["filter_time"] for res in self.eval_results]
        post_retrieve = [res["postfilter"]["retrieve_time"] for res in self.eval_results]
        hybrid_filter = [res["hybrid"]["filter_time"] for res in self.eval_results]
        hybrid_search = [res["hybrid"]["vec_search_time"] for res in self.eval_results]

        width = 0.2
        x = np.arange(len(workloads))

        fig, ax = plt.subplots(2, 2, figsize=(16, 10))

        # --- Recall@K ---
        ax[0, 0].bar(x - width, recall_post, width, label='Postfilter vs Prefilter')
        ax[0, 0].bar(x, recall_hybrid, width, label='Hybrid vs Prefilter')
        ax[0, 0].set_title("Recall@10 per Workload")
        ax[0, 0].set_xlabel("Workload")
        ax[0, 0].set_ylabel("Recall")
        ax[0, 0].set_xticks(x)
        ax[0, 0].set_xticklabels(workload_labels)
        ax[0, 0].set_ylim(0, 1.05)
        ax[0, 0].grid(True)
        ax[0, 0].legend()

        # --- Total Latency ---
        ax[0, 1].bar(x - width, pre_total, width, label='Prefilter Total')
        ax[0, 1].bar(x, post_total, width, label='Postfilter Total')
        ax[0, 1].bar(x + width, hybrid_total, width, label='Hybrid Total')
        ax[0, 1].set_title("Total Latency per Workload")
        ax[0, 1].set_xlabel("Workload")
        ax[0, 1].set_ylabel("Seconds")
        ax[0, 1].set_yscale('log')
        ax[0, 1].set_xticks(x)
        ax[0, 1].set_xticklabels(workload_labels)
        ax[0, 1].legend()
        ax[0, 1].grid(True)

        # --- Breakdown Filter vs Retrieve/Search ---
        ax[1, 0].bar(x - width, pre_filter, width, label='Prefilter Filter')
        ax[1, 0].bar(x, post_filter, width, label='Postfilter Filter')
        ax[1, 0].bar(x + width, hybrid_filter, width, label='Hybrid Filter')
        ax[1, 0].plot(x - width, pre_retrieve, marker='x', linestyle='--', label='Prefilter Retrieve')
        ax[1, 0].plot(x, post_retrieve, marker='x', linestyle='--', label='Postfilter Retrieve')
        ax[1, 0].plot(x + width, hybrid_search, marker='x', linestyle='--', label='Hybrid Vector Search')
        ax[1, 0].set_title("Filter vs Retrieve/Vector Search Time")
        ax[1, 0].set_xlabel("Workload")
        ax[1, 0].set_ylabel("Seconds")
        ax[1, 0].set_yscale('log')
        ax[1, 0].set_xticks(x)
        ax[1, 0].set_xticklabels(workload_labels)
        ax[1, 0].legend()
        ax[1, 0].grid(True)

        # --- Memory Usage ---
        ax[1, 1].bar(x - width, pre_mem, width, label='Prefilter Total Mem')
        ax[1, 1].bar(x, post_mem, width, label='Postfilter Total Mem')
        ax[1, 1].bar(x + width, hybrid_mem, width, label='Hybrid Total Mem')
        ax[1, 1].set_title("Memory Usage per Workload")
        ax[1, 1].set_xlabel("Workload")
        ax[1, 1].set_ylabel("MB")
        ax[1, 1].set_yscale('log')
        ax[1, 1].set_xticks(x)
        ax[1, 1].set_xticklabels(workload_labels)
        ax[1, 1].legend()
        ax[1, 1].grid(True)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300)
            print(f"Plot saved to {save_path}")

        plt.show()

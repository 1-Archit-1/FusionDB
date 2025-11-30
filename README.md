# FusionDB
A high-performance engine for hybrid search, combining semantic vector search with structured attribute filtering.

## Setup and QuickStart

1. Download the dataset to evaluate over. Run the corresponding `embeddings.sh`. This script will download all the data from the LAION dataset into a directory that holds both the metadata and embeddings it will provide.

2. Install dependancies from environment.yml
```bash
conda env create -f environment.yml
```

3. Create a .env file and change the paths as needed 
```bash
cp .env.example .env
```
4. Run the 2 preprocessors for embeddings and metadata
```bash
python3 preprocess_emb.py
python3 preprocess_meta.py
```
5. Once everything is setup, you can test the project.
```python
python3 main.py
``` 
This directly runs the evaluator on both DuckDB based data storage and Index based data storage. If its your first time running this script it will take some time since the FaissIndex and database need to get created and stored to disk, but upon every other instance of running the database, it should be relativley quick to run. 

## Metadata filtering
We have 2 separate implementations for the Relational Metdata Filtering which we compare.

### DuckDB
DuckDB provides SQL-based metadata filtering by querying Parquet files directly from disk. It leverages columnar storage and predicate pushdown for efficient filtering without loading entire datasets into memory. Supports full SQL WHERE clause syntax for flexible query expressions.

**Key Features:**
- Queries Parquet files in-place (no data loading required)
- SQL WHERE clause support: comparisons, ranges, NULL checks, AND/OR logic
- Columnar format enables fast filtering on specific attributes
- In-memory query execution with disk-based data access

### Alternative Metadata Index System
We implemented a custom index-based metadata filtering system as an alternative to DuckDB through specialized indices for different data types.

Custom Index Structures:
  - Range Indices: Sorted arrays with binary search for numerical queries (O(log n))
  - Hash Indices: Dictionary-based lookups for categorical fields (O(1))
  - Two-Stage Filtering: Index-based filtering followed by NumPy array operations for complex conditions

SQL Support:
  - Field-value comparisons: nsfw = 'UNLIKELY', width > 1024
  - Field-field comparisons: width > height, width = height
  - NULL checks: license is NULL, license is not NULL
  - Combined queries: width > height AND license is NULL AND nsfw = 'UNLIKELY'
#### Architecture
```
Query (SQL WHERE clause)
    ↓
SQL Parser
    ↓
Two-Stage Filtering:
    Stage 1: Index-based filtering (e.g. nsfw='UNLIKELY', width>1024)
    Stage 2: Array-based filtering (e.g. width>height, IS NULL checks)
    ↓
Filtered Result IDs
```


## Hybrid Search Algorithm
This hybrid search system combines **FAISS vector indexing** with **metadata filtering** to provide fast and flexible similarity search. It intelligently adapts its strategy based on filter selectivity to optimize performance.
### Performance Optimizations

1. **FAISS-Accelerated Search**: Uses optimized C++ implementations with SIMD vectorization (3-10x faster than numpy brute force)
2. **Memory-Efficient Index Building**: Processes embeddings in batches to avoid loading entire dataset into RAM
3. **Adaptive Strategy Selection**: Automatically chooses between pre-filtering and post-filtering based on selectivity
4. **Compressed Storage**: Uses IVFPQ index (20-50x smaller than raw vectors)

### Adaptive Strategy

The system automatically selects the optimal search strategy:

#### Pre-Filtering (Selective Queries)
- **When**: Filter selectivity < threshold (default 8%)
- **Approach**: Filter metadata first → Search only filtered vectors
- **Best for**: Highly selective queries (e.g., "specific category AND rare attribute")
- **Memory**: Only loads filtered subset into memory

#### Post-Filtering (Broad Queries)  
- **When**: Filter selectivity ≥ threshold
- **Approach**: Search all vectors with FAISS → Filter top-k results
- **Best for**: Broad queries or queries returning large result sets
- **Speed**: Leverages pre-built FAISS index for fast search

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Hybrid Search Flow                      │
└─────────────────────────────────────────────────────────────┘

1. Query Input
   ├─ Vector query (embeddings)
   └─ Metadata filter (SQL WHERE clause)
   
2. Estimate Selectivity
   └─ Run metadata filter to count matching vectors
   
3. Strategy Selection
   │
   ├─ IF selectivity < threshold (e.g., 10%)
   │  └─ PRE-FILTERING:
   │     ├─ Extract filtered vectors from mmap
   │     ├─ Build temporary IndexFlatIP
   │     └─ Search filtered subset
   │
   └─ ELSE
      └─ POST-FILTERING:
         ├─ Search full FAISS index (fast!)
         ├─ Retrieve top-k × 100 candidates
         └─ Filter candidates by metadata
         
4. Return Results
   └─ Top-k vectors matching both vector similarity + metadata
```
### Index Types

#### IVFPQ (Default - Recommended)
- **Memory**: ~8-50MB per 1M vectors (768 dims)
- **Speed**: Fast (search 1M vectors in ~10-50ms)
- **Accuracy**: 90-99% recall (configurable via nprobe)
- **Parameters**:
  - `nlist`: Number of clusters (default: √n)
  - `m`: Sub-quantizers (default: 8)
  - `bits`: Bits per code (default: 8)
  - `nprobe`: Clusters to search (higher = more accurate)

#### IVF (Higher Accuracy)
- **Memory**: ~3GB per 1M vectors (768 dims)
- **Speed**: Moderate
- **Accuracy**: Higher than IVFPQ
- **Use when**: Memory available, need high accuracy

#### Flat (Exact Search)
- **Memory**: ~3GB per 1M vectors (768 dims)  
- **Speed**: Slower (but optimized with SIMD)
- **Accuracy**: 100% (exact)
- **Use when**: Small datasets or need exact results

**Index Size Estimates** (768-dim vectors):

| Vectors | IVFPQ | IVF | Flat |
|---------|-------|-----|------|
| 100K    | ~1MB  | ~300MB | ~300MB |
| 1M      | ~10MB | ~3GB | ~3GB |
| 10M     | ~100MB | ~30GB | ~30GB |

## Evaluation
In order to evaluate our baseline and our hybrid searches, we had to produce an evaluator class that would consist of workloads we deemed necessary to really test our implementation. The evaluator resides in the `evaluator.py` script as a class. You can inspect `main.py` to see how it to use the class, but basically you create the evaluator class specifying one of our metadata filtering implementations (either `'index'` or `'duck'` as mentioned before) to setup the backing database based on the embeddings and metadata located in our directory folder (as we've explained in setup how to install those). 

Given your new Evaluator object, you can invoke the `evaluate()` method to get it to process all of the workloads that were predefined in it and then print the results of those workloads through `export_results()` as text files and even display the graphs through `plot_results()`. 

The evaultor itself tracks the amount of time and usage of memory to run the filterings and vector searches. We compile the results to show the amount of each latency it takes for prefiltering, postfiltering, and hybrid search along with another batch that tests different databases that could be used (Custom indexing structure vs DuckDB). 



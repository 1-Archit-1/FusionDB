# FusionDB
A high-performance engine for hybrid search, combining semantic vector search with structured attribute filtering.

## Setup
First we are going to need the data we will evaluate over. To install the data, run the corresponding `embeddings.sh`. This script will download all the data from the LAION dataset into a directory that holds both the metadata and embeddings it will provide. Then you just need to setup a conda environment that will have [faiss](https://github.com/facebookresearch/faiss/blob/main/INSTALL.md) installed along with `psutil`, `sentence_transformers`, `PIL`, `torch` so that the evaluation logic works correctly.

Once everything is setup, you can just test the project by running `main.py` to directly run the evaluator on both DuckDB based data storage and Index based data storage. If its your first time running this script it will take some time since the FaissIndex and database need to get created and stored to disk, but upon every other instance of running the database, it should be relativley quick to run. 

## Metadata filtering


### Alternative Metadata Index System
Custom index for metadata filtering, serving as an alternative to DuckDB for hybrid search queries through specialized indices for different data types:
  - Range Indices: Binary search on sorted arrays for numeric fields (O(log n) query time)
  - Hash Indices: Dictionary-based lookup for categorical fields (O(1) query time)
  - Field-vs-Field Support: NumPy vectorized operations for complex comparisons
  - SQL Compatibility: Support parser supporting NULL checks and variable comparisons

Architecture：
Query (SQL WHERE clause)
    ↓
SQL Parser
    ↓
Two-Stage Filtering:
    Stage 1: Index-based filtering (e.g.nsfw='UNLIKELY', width>1024)
    Stage 2: Array-based filtering (e.g.width>height, IS NULL checks)
    ↓
Filtered Result IDs

Usage

## Hybrid Search Algorithm

## Evaluation
In order to evaluate our baseline and our hybrid searches, we had to produce an evaluator class that would consist of workloads we deemed necessary to really test our implementation. The evaluator resides in the `evaluator.py` script as a class. You can inspect `main.py` to see how it to use the class, but basically you create the evaluator class specifying one of our metadata filtering implementations (either `'index'` or `'duck'` as mentioned before) to setup the backing database based on the embeddings and metadata located in our directory folder (as we've explained in setup how to install those). 

Given your new Evaluator object, you can invoke the `evaluate()` method to get it to process all of the workloads that were predefined in it and then print the results of those workloads through `export_results()` as text files and even display the graphs through `plot_results()`. 

The evaultor itself tracks the amount of time and usage of memory to run the filterings and vector searches. We compile the results to show the amount of each latency it takes for prefiltering, postfiltering, and hybrid search along with another batch that tests different databases that could be used (Custom indexing structure vs DuckDB). 

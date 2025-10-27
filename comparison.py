import pandas as pd
import numpy as np
import duckdb
import time
from metadata_index_manager import IndexManager

def benchmark_all(parquet_file='metadata_1.parquet'):
    """
    Benchmark: Your Index vs Pandas vs DuckDB
    """
    
    print("="*80)
    print("Performance Comparison: Your Index vs Pandas vs DuckDB")
    print("="*80)
    
    # ========================================================================
    # Preparation
    # ========================================================================
    
    print("\n[Preparation]")
    print("-" * 80)
    
    # 1. Load original DataFrame (needed for Pandas)
    print("Loading Pandas DataFrame...")
    start = time.time()
    df = pd.read_parquet(parquet_file)
    pandas_load_time = time.time() - start
    print(f"  Time: {pandas_load_time:.2f}s")
    
    # 2. Your index sys
    print("\nBuilding your index system...")
    start = time.time()
    mgr = IndexManager(parquet_file).build()
    your_build_time = time.time() - start
    print(f"  Time: {your_build_time:.2f}s")
    
    # 3. DuckDB
    print("\nInitializing DuckDB...")
    start = time.time()
    conn = duckdb.connect(':memory:')
    conn.execute(f"""
        CREATE TABLE metadata AS 
        SELECT * FROM read_parquet('{parquet_file}')
    """)
    duckdb_load_time = time.time() - start
    print(f"  Time: {duckdb_load_time:.2f}s")
    
    print("\n" + "="*80)
    print("Starting benchmark tests (run 3 times and average)")
    print("="*80)
    
    results = []
    
    # ========================================================================
    # Scenario 1: Simple filter - NSFW
    # ========================================================================
    print("\n[Scenario 1] Simple filter: NSFW = 'UNLIKELY'")
    print("-" * 80)
    
    # Your index
    print("\n[Your Index]")
    times = []
    for _ in range(3):
        start = time.time()
        r = mgr.query(nsfw='UNLIKELY')
        times.append(time.time() - start)
    your_time = np.mean(times) * 1000
    your_count = len(r)
    print(f"  Time: {your_time:.2f}ms")
    print(f"  Result: {your_count:,} rows")
    
    # Pandas
    print("\n[Pandas]")
    times = []
    for _ in range(3):
        start = time.time()
        r = df[df['NSFW'] == 'UNLIKELY']
        times.append(time.time() - start)
    pandas_time = np.mean(times) * 1000
    pandas_count = len(r)
    print(f"  Time: {pandas_time:.2f}ms")
    print(f"  Result: {pandas_count:,} rows")
    
    # DuckDB
    print("\n[DuckDB]")
    times = []
    for _ in range(3):
        start = time.time()
        r = conn.execute("SELECT * FROM metadata WHERE NSFW = 'UNLIKELY'").fetchdf()
        times.append(time.time() - start)
    duckdb_time = np.mean(times) * 1000
    duckdb_count = len(r)
    print(f"  Time: {duckdb_time:.2f}ms")
    print(f"  Result: {duckdb_count:,} rows")
    
    print(f"\n Speedup:")
    print(f"  vs Pandas: {pandas_time/your_time:.1f}x")
    print(f"  vs DuckDB: {duckdb_time/your_time:.1f}x")
    
    results.append({
        'Scenario': 'Simple filter',
        'Your Index': your_time,
        'Pandas': pandas_time,
        'DuckDB': duckdb_time,
        'vs_Pandas': pandas_time/your_time,
        'vs_DuckDB': duckdb_time/your_time
    })
    
    # ========================================================================
    # Scenario 2: Range query
    # ========================================================================
    print("\n[Scenario 2] Range query: similarity > 0.35")
    print("-" * 80)
    
    # Your index
    print("\n[Your Index]")
    times = []
    for _ in range(3):
        start = time.time()
        r = mgr.query(similarity_min=0.35)
        times.append(time.time() - start)
    your_time = np.mean(times) * 1000
    your_count = len(r)
    print(f"  Time: {your_time:.2f}ms")
    print(f"  Result: {your_count:,} rows")
    
    # Pandas
    print("\n[Pandas]")
    times = []
    for _ in range(3):
        start = time.time()
        r = df[df['similarity'] > 0.35]
        times.append(time.time() - start)
    pandas_time = np.mean(times) * 1000
    pandas_count = len(r)
    print(f"  Time: {pandas_time:.2f}ms")
    print(f"  Result: {pandas_count:,} rows")
    
    # DuckDB
    print("\n[DuckDB]")
    times = []
    for _ in range(3):
        start = time.time()
        r = conn.execute("SELECT * FROM metadata WHERE similarity > 0.35").fetchdf()
        times.append(time.time() - start)
    duckdb_time = np.mean(times) * 1000
    duckdb_count = len(r)
    print(f"  Time: {duckdb_time:.2f}ms")
    print(f"  Result: {duckdb_count:,} rows")
    
    print(f"\n Speedup:")
    print(f"  vs Pandas: {pandas_time/your_time:.1f}x")
    print(f"  vs DuckDB: {duckdb_time/your_time:.1f}x")
    
    results.append({
        'Scenario': 'Range query',
        'Your Index': your_time,
        'Pandas': pandas_time,
        'DuckDB': duckdb_time,
        'vs_Pandas': pandas_time/your_time,
        'vs_DuckDB': duckdb_time/your_time
    })
    
    # ========================================================================
    # Scenario 3: Multi-condition query
    # ========================================================================
    print("\n[Scenario 3] Multi-condition: NSFW + similarity + width")
    print("-" * 80)
    
    # Your index
    print("\n[Your Index]")
    times = []
    for _ in range(3):
        start = time.time()
        r = mgr.query(
            nsfw='UNLIKELY',
            similarity_min=0.35,
            width_min=500
        )
        times.append(time.time() - start)
    your_time = np.mean(times) * 1000
    your_count = len(r)
    print(f"  Time: {your_time:.2f}ms")
    print(f"  Result: {your_count:,} rows")
    
    # Pandas
    print("\n[Pandas]")
    times = []
    for _ in range(3):
        start = time.time()
        r = df[
            (df['NSFW'] == 'UNLIKELY') &
            (df['similarity'] > 0.35) &
            (df['original_width'] > 500)
        ]
        times.append(time.time() - start)
    pandas_time = np.mean(times) * 1000
    pandas_count = len(r)
    print(f"  Time: {pandas_time:.2f}ms")
    print(f"  Result: {pandas_count:,} rows")
    
    # DuckDB
    print("\n[DuckDB]")
    times = []
    for _ in range(3):
        start = time.time()
        r = conn.execute("""
            SELECT * FROM metadata 
            WHERE NSFW = 'UNLIKELY' 
              AND similarity > 0.35 
              AND original_width > 500
        """).fetchdf()
        times.append(time.time() - start)
    duckdb_time = np.mean(times) * 1000
    duckdb_count = len(r)
    print(f"  Time: {duckdb_time:.2f}ms")
    print(f"  Result: {duckdb_count:,} rows")
    
    print(f"\n Speedup:")
    print(f"  vs Pandas: {pandas_time/your_time:.1f}x")
    print(f"  vs DuckDB: {duckdb_time/your_time:.1f}x")
    
    results.append({
        'Scenario': 'Multi-condition',
        'Your Index': your_time,
        'Pandas': pandas_time,
        'DuckDB': duckdb_time,
        'vs_Pandas': pandas_time/your_time,
        'vs_DuckDB': duckdb_time/your_time
    })
    
    # ========================================================================
    # Scenario 4: Text search (your strong point!)
    # ========================================================================
    print("\n[Scenario 4] Text search: caption contains 'woman'")
    print("-" * 80)
    
    # Your index
    print("\n[Your Index - inverted index]")
    times = []
    for _ in range(3):
        start = time.time()
        r = mgr.query_caption('woman')
        times.append(time.time() - start)
    your_time = np.mean(times) * 1000
    your_count = len(r)
    print(f"  Time: {your_time:.2f}ms")
    print(f"  Result: {your_count:,} rows")
    
    # Pandas
    print("\n[Pandas - str.contains]")
    times = []
    for _ in range(3):
        start = time.time()
        r = df[df['caption'].str.contains('woman', case=False, na=False)]
        times.append(time.time() - start)
    pandas_time = np.mean(times) * 1000
    pandas_count = len(r)
    print(f"  Time: {pandas_time:.2f}ms")
    print(f"  Result: {pandas_count:,} rows")
    
    # DuckDB
    print("\n[DuckDB - LIKE]")
    times = []
    for _ in range(3):
        start = time.time()
        r = conn.execute("""
            SELECT * FROM metadata 
            WHERE caption LIKE '%woman%'
        """).fetchdf()
        times.append(time.time() - start)
    duckdb_time = np.mean(times) * 1000
    duckdb_count = len(r)
    print(f"  Time: {duckdb_time:.2f}ms")
    print(f"  Result: {duckdb_count:,} rows")
    
    print(f"\n Speedup:")
    print(f"  vs Pandas: {pandas_time/your_time:.1f}x")
    print(f"  vs DuckDB: {duckdb_time/your_time:.1f}x")
    print(f"  → This is your biggest advantage! Inverted index >> full scan")
    
    results.append({
        'Scenario': 'Text search',
        'Your Index': your_time,
        'Pandas': pandas_time,
        'DuckDB': duckdb_time,
        'vs_Pandas': pandas_time/your_time,
        'vs_DuckDB': duckdb_time/your_time
    })

    # ========================================================================
    # Scenario 5: Complex mixed query
    # ========================================================================
    print("\n[Scenario 5] Complex mixed query: NSFW + similarity + width + caption")
    print("-" * 80)
    
    # Your index
    print("\n[Your Index]")
    times = []
    for _ in range(3):
        start = time.time()
        r = mgr.query(
            nsfw='UNLIKELY',
            similarity_min=0.35,
            width_min=500,
            caption_contains='woman'
        )
        times.append(time.time() - start)
    your_time = np.mean(times) * 1000
    your_count = len(r)
    print(f"  Time: {your_time:.2f}ms")
    print(f"  Result: {your_count:,} rows")
    
    # Pandas
    print("\n[Pandas]")
    times = []
    for _ in range(3):
        start = time.time()
        r = df[
            (df['NSFW'] == 'UNLIKELY') &
            (df['similarity'] > 0.35) &
            (df['original_width'] > 500) &
            (df['caption'].str.contains('woman', case=False, na=False))
        ]
        times.append(time.time() - start)
    pandas_time = np.mean(times) * 1000
    pandas_count = len(r)
    print(f"  Time: {pandas_time:.2f}ms")
    print(f"  Result: {pandas_count:,} rows")
    
    # DuckDB
    print("\n[DuckDB]")
    times = []
    for _ in range(3):
        start = time.time()
        r = conn.execute("""
            SELECT * FROM metadata
            WHERE NSFW = 'UNLIKELY'
              AND similarity > 0.35
              AND original_width > 500
              AND caption LIKE '%woman%'
        """).fetchdf()
        times.append(time.time() - start)
    duckdb_time = np.mean(times) * 1000
    duckdb_count = len(r)
    print(f"  Time: {duckdb_time:.2f}ms")
    print(f"  Result: {duckdb_count:,} rows")
    
    print(f"\n Speedup:")
    print(f"  vs Pandas: {pandas_time/your_time:.1f}x")
    print(f"  vs DuckDB: {duckdb_time/your_time:.1f}x")
    
    results.append({
        'Scenario': 'Complex mixed query',
        'Your Index': your_time,
        'Pandas': pandas_time,
        'DuckDB': duckdb_time,
        'vs_Pandas': pandas_time/your_time,
        'vs_DuckDB': duckdb_time/your_time
    })
    
    # ========================================================================
    # Summary Table
    # ========================================================================
    print("\n" + "="*80)
    print("Benchmark Summary (average times in ms)")
    print("="*80)
    
    df_results = pd.DataFrame(results)
    print(df_results[['Scenario', 'Your Index', 'Pandas', 'DuckDB', 'vs_Pandas', 'vs_DuckDB']])
    
    # Optional: Save results to CSV
    df_results.to_csv('benchmark_results.txt', index=False)
    print("\n Results saved to 'benchmark_results.txt'")
    
    print("\nAll benchmarks completed!")
    print("="*80)

# ========================================================================
# Run benchmark
# ========================================================================
if __name__ == "__main__":
    benchmark_all(parquet_file='metadata_1.parquet')

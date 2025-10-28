import numpy as np
import pandas as pd
import duckdb
import glob

def db_implementation(meta_path):
    """
    Sets up DuckDB to read metadata from Parquet files using a glob pattern.
    Returns the DuckDB connection object.
    """
    print("Initializing in-memory DuckDB instance...")
    con = duckdb.connect(database=':memory:', read_only=False)
    parquet_files = sorted(glob.glob(meta_path))
    if not parquet_files:
        print(f"Error: No metadata files found matching {meta_path}")
        exit()

    print(f"DuckDB will query the following files from disk: {parquet_files}")

    try:
        print("\nMetadata Schema (from first file):")
        con.sql(f"DESCRIBE SELECT * FROM read_parquet('{parquet_files[0]}')").show()
    except Exception as e:
        print(f"Error reading schema from {parquet_files[0]}: {e}")
        exit()
    # Get total row count from metadata files for validation
    try:
        total_metadata_rows = con.sql(f"SELECT COUNT(*) FROM read_parquet('{meta_path}')").fetchone()[0]
        print(f"Found {total_metadata_rows} total rows in metadata files.")
    except Exception as e:
        print(f"Error getting metadata row count: {e}")
        exit()

    return { 'con': con, 'total_metadata_rows': total_metadata_rows, 'meta_path': meta_path }

def run_query(res, where_clause):
    """
    Executes a query on the DuckDB connection and returns filtered IDs.
    """
    con = res['con']
    total_metadata_rows = res['total_metadata_rows']
    meta_path = res['meta_path']
    parquet_files = sorted(glob.glob(meta_path))
    try:
        query = f"""
        SELECT (row_number() over ()) - 1 as id 
        FROM read_parquet({parquet_files}, filename=true)
        WHERE {where_clause}
        """
        filtered_ids_result = con.execute(query).fetchall()
        filtered_ids = np.array([item[0] for item in filtered_ids_result])
        print(f"Query returned {len(filtered_ids)} rows out of {total_metadata_rows} total rows.")
        return filtered_ids
    except Exception as e:
        print(f"Error executing query: {e}")
        return np.array([])


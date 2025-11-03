#############################################################################
## This script processes metadata parquet files by adding an index column.
## It saves the modified files to a specified output directory.
#############################################################################

import pandas as pd
import glob
import os
# load .env
from dotenv import load_dotenv
config = load_dotenv()

pq_dir = os.getenv('raw_meta_path')
output_dir = os.getenv('meta_path')

# look for all parquet files in the directory matching metadata_*.parquet

parquet_files = sorted(glob.glob(os.path.join(pq_dir, "metadata_*.parquet")))
current_index = 0

for file in parquet_files:
    print(f"Processing {file}...")
    df = pd.read_parquet(file)
    df['index'] = range(current_index, current_index + len(df))
    current_index += len(df)
    output_file = os.path.join(output_dir, os.path.basename(file))
    df.to_parquet(output_file)
    print(f"Saved with index to {output_file}")


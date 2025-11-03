############################
## This script concatenates multiple NumPy embedding files into a single large memory-mapped file.
## It is designed to handle large datasets without loading everything into RAM.
############################


import numpy as np
import glob
import os
from numpy.lib.format import open_memmap
from dotenv import load_dotenv

# load .env
config = load_dotenv()
# --- 1. Configuration ---
embeddings_dir = os.getenv('embeddings_dir_path')
EMBEDDINGS_GLOB_PATH = f'{embeddings_dir}/text_emb_*.npy'
OUTPUT_FILENAME = os.getenv('embeddings_path')
# We'll assume float32, which is standard for embeddings
DTYPE = np.float32 

print(f"Finding files matching {EMBEDDINGS_GLOB_PATH}...")
embedding_files = sorted(glob.glob(EMBEDDINGS_GLOB_PATH))
if not embedding_files:
    print(f"Error: No embedding files found matching {EMBEDDINGS_GLOB_PATH}.")
    exit()

print(f"Found {len(embedding_files)} files to concatenate.")

# --- 2. Determine total shape ---
total_rows = 0
dimension = -1

for f in embedding_files:
    print(f"Probing {f}...")
    try:
        # Load file in mmap_mode to just read its shape
        data = np.load(f, mmap_mode='r')
        rows, d = data.shape
        total_rows += rows
        
        if dimension == -1:
            dimension = d
        elif dimension != d:
            print(f"Error: Mismatched dimensions! {f} has {d}, expected {dimension}")
            exit()
            
    except Exception as e:
        print(f"Error reading {f}: {e}")
        exit()

print(f"\nTotal vectors: {total_rows}")
print(f"Vector dimension: {dimension}")
print(f"Output file: {OUTPUT_FILENAME}")

if os.path.exists(OUTPUT_FILENAME):
    print(f"Warning: Output file {OUTPUT_FILENAME} already exists. Deleting it.")
    os.remove(OUTPUT_FILENAME)

# --- 3. Create the empty destination file ---
# This creates a new, empty file on disk with the correct total shape.
# It doesn't use RAM.
print(f"Creating empty destination file with shape ({total_rows}, {dimension})...")
dest_mmap = open_memmap(
    OUTPUT_FILENAME,
    mode='w+',
    dtype=DTYPE,
    shape=(total_rows, dimension)
)

# --- 4. Loop, load, and write ---
current_row_index = 0
try:
    for f in embedding_files:
        print(f"Processing {f}...")
        try:
            # Memory-map the source file
            source_data = np.load(f, mmap_mode='r')
            rows, d = source_data.shape

            # Write in chunks (e.g., 10,000 rows at a time)
            chunk_size = 10000
            for start in range(0, rows, chunk_size):
                end = min(start + chunk_size, rows)
                chunk = source_data[start:end].astype(DTYPE)
                dest_mmap[current_row_index : current_row_index + (end - start)] = chunk
                current_row_index += (end - start)

            print(f"  ... wrote {rows} vectors. Total written: {current_row_index}/{total_rows}")

        except Exception as e:
            print(f"Error processing {f}: {e}")
            exit()
except Exception as e:
    print(f"Unexpected error during processing: {e}")
finally:
    # clean up 
    del dest_mmap


# --- 5. Finalize ---
# Flush all changes to disk
# dest_mmap.flush()
# del dest_mmap # Close the memory-mapped file
print("\nDone!")
print(f"All embeddings successfully concatenated into {OUTPUT_FILENAME}.")
print(f"You can now update baseline_prefiltering.py to use this file.")

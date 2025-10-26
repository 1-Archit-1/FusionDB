import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer

encoder = SentenceTransformer("distiluse-base-multilingual-cased-v2")

df = pd.read_parquet("metadata_1.parquet")

# Data Related
# img_embeddings = np.load('img_emb_1.npy')
text_embeddings = np.load('text_emb_1.npy').astype(np.float32)
d = text_embeddings.shape[1]
nb = text_embeddings.shape[0]
nq = 0 # Number of queries but we haven't determined that yet

index = faiss.IndexFlatL2(d)
faiss.normalize_L2(text_embeddings)
index.add(text_embeddings)

search_text = 'Fundamental'
search_vector = encoder.encode(search_text)
_vector = np.array([search_vector]).astype(np.float32)
faiss.normalize_L2(_vector)

k = index.ntotal
D, I = index.search(_vector, k)

print(D)
print(I)

print(f"Search resulrsts for: '{search_text}'")
for rank, idx in enumerate(I[0]):
    print(f"Rank {rank+1}: {df.iloc[idx]['caption']}")
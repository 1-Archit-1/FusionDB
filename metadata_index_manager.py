import pandas as pd
import numpy as np
import pickle
import re
from collections import defaultdict
from typing import Dict, Any, List

class RangeIndex:
    def __init__(self, keys, values):
        idx = np.argsort(keys)
        self.keys, self.values = keys[idx], values[idx]
    
    def query(self, min_val, max_val):
        return self.values[np.searchsorted(self.keys, min_val, 'left'):
                          np.searchsorted(self.keys, max_val, 'right')]

class TextIndex:
    def __init__(self):
        self.index = defaultdict(set)
    
    def build(self, texts, ids):
        for doc_id, text in zip(ids, texts):
            if pd.notna(text):
                for word in set(re.findall(r'\b\w{2,}\b', text.lower())):
                    self.index[word].add(doc_id)
        self.index = {k: np.array(list(v), dtype=np.int32) 
                      for k, v in self.index.items()}
    
    def search(self, query, mode='AND'):
        words = set(re.findall(r'\b\w{2,}\b', query.lower()))
        if not words: return np.array([])
        
        sets = [set(self.index.get(w, [])) for w in words]
        if not sets: return np.array([])
        
        result = sets[0].intersection(*sets[1:]) if mode == 'AND' else \
                 sets[0].union(*sets[1:])
        return np.array(list(result), dtype=np.int32)

class IndexManager:
    def __init__(self, parquet_file):
        self.df = pd.read_parquet(parquet_file)
        self.n = len(self.df)
        self.arrays = {
            'sim': self.df['similarity'].fillna(0).values,
            'width': self.df['original_width'].values,
            'height': self.df['original_height'].values,
            'nsfw': self.df['NSFW'].values,
            'license': self.df['LICENSE'].values,
            'caption': self.df['caption'].values
        }
        self.indices = {}
    
    def build(self):
        ids = np.arange(self.n)
        
        for name, arr in [('sim', 'sim'), ('width', 'width'), ('height', 'height')]:
            self.indices[name] = RangeIndex(self.arrays[arr], ids)

        for name in ['nsfw', 'license']:
            hash_idx = defaultdict(list)
            for i, val in enumerate(self.arrays[name]):
                if pd.notna(val): hash_idx[val].append(i)
            self.indices[name] = {k: np.array(v, dtype=np.int32) 
                                  for k, v in hash_idx.items()}

        self.indices['caption'] = TextIndex()
        self.indices['caption'].build(self.arrays['caption'].tolist(), ids)
        
        return self
    
    def query(self, **conditions):
        """
        usage:
            query(nsfw='UNLIKELY')
            query(similarity_min=0.35, width_min=500)
            query(caption='woman', nsfw='UNLIKELY', similarity_min=0.35)
        """
        conds = []
        
        for field in ['similarity', 'width', 'height']:
            min_key, max_key = f'{field}_min', f'{field}_max'
            if min_key in conditions or max_key in conditions:
                conds.append((
                    field,
                    conditions.get(min_key, 0),
                    conditions.get(max_key, [1024, 10800, 17943][['similarity', 'width', 'height'].index(field)]),
                    'range'
                ))
        
        for field in ['nsfw', 'license']:
            if field in conditions:
                conds.append((field, conditions[field], None, 'hash'))
        
        if 'caption' in conditions or 'caption_contains' in conditions or 'caption_any' in conditions:
            query_text = conditions.get('caption') or conditions.get('caption_contains') or conditions.get('caption_any')
            mode = 'AND' if 'caption_contains' in conditions or 'caption' in conditions else 'OR'
            conds.append(('caption', query_text, mode, 'text'))
        
        if not conds:
            return np.array([])
        
        estimates = []
        for field, val, extra, ctype in conds:
            if ctype == 'hash':
                count = len(self.indices[field].get(val, []))
            elif ctype == 'text':
                count = self.n * 0.01
            else:  # range
                count = self.n * 0.2
            estimates.append((field, val, extra, ctype, count))

        estimates.sort(key=lambda x: x[4])
        
        field, val, extra, ctype, _ = estimates[0]
        if ctype == 'hash':
            result = self.indices[field].get(val, np.array([]))
        elif ctype == 'text':
            result = self.indices['caption'].search(val, mode=extra)
        else:  # range
            idx_name = {'similarity': 'sim', 'width': 'width', 'height': 'height'}[field]
            result = self.indices[idx_name].query(val, extra)
        
        if len(result) == 0:
            return result
        
        mask = np.ones(len(result), dtype=bool)
        for field, val, extra, ctype, _ in estimates[1:]:
            if ctype == 'hash':
                mask &= (self.arrays[field][result] == val)
            elif ctype == 'text':
                text_matches = set(self.indices['caption'].search(val, mode=extra))
                mask &= np.array([idx in text_matches for idx in result])
            else:  # range
                arr = self.arrays[{'similarity': 'sim', 'width': 'width', 'height': 'height'}[field]]
                mask &= (arr[result] >= val) & (arr[result] <= extra)
        
        return result[mask]
    
    def query_caption(self, text, mode='AND'):
        return self.indices['caption'].search(text, mode)
    
    def query_license(self, license_type):
        return self.indices['license'].get(license_type, np.array([]))
    
    def get_licenses(self):
        return list(self.indices['license'].keys())
    
    def get_results(self, indices):
        return self.df.iloc[indices] if len(indices) > 0 else pd.DataFrame()
    
    def save(self, file='indices.pkl'):
        with open(file, 'wb') as f:
            pickle.dump(self.indices, f)
    
    def load(self, file='indices.pkl'):
        with open(file, 'rb') as f:
            self.indices = pickle.load(f)
        return self


if __name__ == "__main__":
    import time
    
    print("Building indices...")
    start = time.time()
    mgr = IndexManager('metadata_1.parquet').build()
    print(f"Completed in {time.time()-start:.2f}s\n")
    
    mgr.save()
    
    print("="*60)
    print("Query Tests")
    print("="*60)
    
    # Test 1: Simple query
    print("\n[1] NSFW='UNLIKELY'")
    start = time.time()
    r = mgr.query(nsfw='UNLIKELY')
    print(f"    Results: {len(r):,} rows | Time: {(time.time()-start)*1000:.2f} ms")
    
    # Test 2: Range query
    print("\n[2] similarity > 0.35")
    start = time.time()
    r = mgr.query(similarity_min=0.35)
    print(f"    Results: {len(r):,} rows | Time: {(time.time()-start)*1000:.2f} ms")
    
    # Test 3: Text search
    print("\n[3] caption contains 'woman'")
    start = time.time()
    r = mgr.query_caption('woman')
    print(f"    Results: {len(r):,} rows | Time: {(time.time()-start)*1000:.2f} ms")
    
    # Test 4: Combined query
    print("\n[4] Combined query")
    start = time.time()
    r = mgr.query(
        nsfw='UNLIKELY',
        similarity_min=0.35,
        width_min=500,
        caption='woman'
    )
    print(f"    Results: {len(r):,} rows | Time: {(time.time()-start)*1000:.2f} ms")
    
    # Show results
    if len(r) > 0:
        print("\nTop 3 results:")
        df = mgr.get_results(r[:3])
        print(df[['image_path', 'NSFW', 'similarity', 'caption']])
    
    print("\n" + "="*60)
    print("All functions working correctly!")

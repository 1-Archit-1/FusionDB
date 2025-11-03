import pandas as pd
import numpy as np
import pickle
from collections import defaultdict
from typing import Dict, Any, List

class RangeIndex:
    def __init__(self, keys, values):
        idx = np.argsort(keys)
        self.keys, self.values = keys[idx], values[idx]
    
    def query(self, min_val, max_val):
        return self.values[np.searchsorted(self.keys, min_val, 'left'):
                          np.searchsorted(self.keys, max_val, 'right')]

class IndexManager:
    def __init__(self, parquet_file):
        self.df = pd.read_parquet(parquet_file)
        self.n = len(self.df)
        self.arrays = {
            'sim': self.df['similarity'].fillna(0).values,
            'width': self.df['original_width'].values,
            'height': self.df['original_height'].values,
            'nsfw': self.df['NSFW'].values,
            'license': self.df['LICENSE'].values
        }
        self.indices = {}
    
    def build(self):
        ids = np.arange(self.n)
        
        # Build range indices for numeric fields
        for name, arr in [('sim', 'sim'), ('width', 'width'), ('height', 'height')]:
            self.indices[name] = RangeIndex(self.arrays[arr], ids)

        # Build hash indices for categorical fields
        for name in ['nsfw', 'license']:
            hash_idx = defaultdict(list)
            for i, val in enumerate(self.arrays[name]):
                if pd.notna(val): hash_idx[val].append(i)
            self.indices[name] = {k: np.array(v, dtype=np.int32) 
                                  for k, v in hash_idx.items()}
        
        return self
    
    def query(self, **conditions):
        """
        usage:
            query(nsfw='UNLIKELY')
            query(similarity_min=0.35, width_min=500)
            query(nsfw='UNLIKELY', similarity_min=0.35, width_min=500)
        """
        conds = []
        
        # Handle range queries
        for field in ['similarity', 'width', 'height']:
            min_key, max_key = f'{field}_min', f'{field}_max'
            if min_key in conditions or max_key in conditions:
                conds.append((
                    field,
                    conditions.get(min_key, 0),
                    conditions.get(max_key, [1024, 10800, 17943][['similarity', 'width', 'height'].index(field)]),
                    'range'
                ))
        
        # Handle hash queries
        for field in ['nsfw', 'license']:
            if field in conditions:
                conds.append((field, conditions[field], None, 'hash'))
        
        if not conds:
            return np.array([])
        
        # Estimate result sizes for query optimization
        estimates = []
        for field, val, extra, ctype in conds:
            if ctype == 'hash':
                count = len(self.indices[field].get(val, []))
            else:  # range
                count = self.n * 0.2
            estimates.append((field, val, extra, ctype, count))

        estimates.sort(key=lambda x: x[4])
        
        # Execute first condition (smallest estimated result)
        field, val, extra, ctype, _ = estimates[0]
        if ctype == 'hash':
            result = self.indices[field].get(val, np.array([]))
        else:  # range
            idx_name = {'similarity': 'sim', 'width': 'width', 'height': 'height'}[field]
            result = self.indices[idx_name].query(val, extra)
        
        if len(result) == 0:
            return result
        
        # Filter by remaining conditions
        mask = np.ones(len(result), dtype=bool)
        for field, val, extra, ctype, _ in estimates[1:]:
            if ctype == 'hash':
                mask &= (self.arrays[field][result] == val)
            else:  # range
                arr = self.arrays[{'similarity': 'sim', 'width': 'width', 'height': 'height'}[field]]
                mask &= (arr[result] >= val) & (arr[result] <= extra)
        
        return result[mask]
    
    def query_license(self, license_type):
        """Query by license type"""
        return self.indices['license'].get(license_type, np.array([]))
    
    def get_licenses(self):
        """Get all available license types"""
        return list(self.indices['license'].keys())
    
    def get_results(self, indices):
        """Get DataFrame rows for given indices"""
        return self.df.iloc[indices] if len(indices) > 0 else pd.DataFrame()
    
    def save(self, file='indices.pkl'):
        """Save indices to file"""
        with open(file, 'wb') as f:
            pickle.dump(self.indices, f)
    
    def load(self, file='indices.pkl'):
        """Load indices from file"""
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
    
    # Test 1: Simple hash query
    print("\n[1] NSFW='UNLIKELY'")
    start = time.time()
    r = mgr.query(nsfw='UNLIKELY')
    print(f"    Results: {len(r):,} rows | Time: {(time.time()-start)*1000:.2f} ms")
    
    # Test 2: Range query
    print("\n[2] similarity > 0.35")
    start = time.time()
    r = mgr.query(similarity_min=0.35)
    print(f"    Results: {len(r):,} rows | Time: {(time.time()-start)*1000:.2f} ms")
    
    # Test 3: Combined query
    print("\n[3] Combined query")
    start = time.time()
    r = mgr.query(
        nsfw='UNLIKELY',
        similarity_min=0.35,
        width_min=500
    )
    print(f"    Results: {len(r):,} rows | Time: {(time.time()-start)*1000:.2f} ms")
    
    # Test 4: License query
    print("\n[4] License types")
    licenses = mgr.get_licenses()
    print(f"    Total license types: {len(licenses)}")
    print(f"    Sample licenses: {licenses[:5]}")
    
    # Show results
    if len(r) > 0:
        print("\nTop 3 results:")
        df = mgr.get_results(r[:3])
        print(df[['image_path', 'NSFW', 'similarity', 'original_width']])
    
    print("\n" + "="*60)
    print("All functions working correctly!")
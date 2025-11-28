"""
IndexManager implementation for metadata filtering
Alternative to DuckDB using custom tree-based indices
"""

import numpy as np
import pandas as pd
import glob
from metadata_index_manager import IndexManager

class IndexedMetadataFilter:
    """
    Provides the same interface as duckdb_rel but uses IndexManager
    """
    
    def __init__(self):
        self.managers = {}  # Cache for IndexManagers
        self.file_ranges = []
        self.total_rows = 0
    
    def db_implementation(self, meta_path):
        """
        Initialize IndexManagers for all metadata files
        (Equivalent to duckdb_rel.db_implementation)
        
        Args:
            meta_path: Glob pattern like 'data/metadata_*.parquet'
        
        Returns:
            dict: Contains managers, total_rows, and meta_path
        """
        print("Initializing IndexManager-based metadata system...")
        
        parquet_files = sorted(glob.glob(meta_path))
        if not parquet_files:
            print(f"Error: No metadata files found matching {meta_path}")
            exit()
        
        print(f"Found {len(parquet_files)} metadata files:")
        for f in parquet_files:
            print(f"  - {f}")
        
        # Build index for each file and track global ranges
        global_offset = 0
        
        for file_path in parquet_files:
            print(f"\nBuilding indices for {file_path}...")
            
            # Create and build IndexManager
            mgr = IndexManager(file_path).build()
            
            # Store manager
            self.managers[file_path] = mgr
            
            # Track global index range
            self.file_ranges.append({
                'file': file_path,
                'manager': mgr,
                'global_start': global_offset,
                'global_end': global_offset + mgr.n - 1,
                'size': mgr.n
            })
            
            global_offset += mgr.n
            self.total_rows += mgr.n
            
            print(f"  ✓ Built indices for {mgr.n:,} rows")
            print(f"  Global range: {self.file_ranges[-1]['global_start']}-{self.file_ranges[-1]['global_end']}")
        
        print(f"\nTotal metadata rows: {self.total_rows:,}")
        
        return {
            'filter_obj': self,
            'total_metadata_rows': self.total_rows,
            'meta_path': meta_path
        }
    
    def run_query(self, res, sql_where_clause):
        """
        Execute query using IndexManager
        (Equivalent to duckdb_rel.run_query)
        
        Args:
            res: Result from db_implementation
            sql_where_clause: SQL WHERE clause (will be parsed)
        
        Returns:
            np.array: Global indices of matching rows
        """
        print(f"Parsing SQL filter: {sql_where_clause}")
        
        # Parse SQL clause into IndexManager conditions
        conditions = self._parse_sql_to_conditions(sql_where_clause)
        
        print(f"Parsed conditions: {conditions}")
        
        # Query each file and collect results
        all_results = []
        
        for file_range in self.file_ranges:
            mgr = file_range['manager']
            
            # Query this file
            local_indices = mgr.query(**conditions)
            
            # Convert to global indices
            if len(local_indices) > 0:
                global_indices = local_indices + file_range['global_start']
                all_results.extend(global_indices)
        
        result_array = np.array(all_results, dtype=np.int64)
        
        print(f"Query returned {len(result_array)} rows out of {self.total_rows} total rows.")
        
        return result_array
    
    def _parse_sql_to_conditions(self, sql_clause):
        """
        Parse SQL WHERE clause into IndexManager query conditions
        
        Examples:
            "NSFW == 'UNLIKELY'" → {'nsfw': 'UNLIKELY'}
            "original_width > 1024" → {'width_min': 1024}
            "similarity > 0.3 and original_width > 800" → 
                {'similarity_min': 0.3, 'width_min': 800}
        """
        conditions = {}
        
        # Clean up the clause
        sql_clause = sql_clause.strip()
        
        # Split by 'and' or 'AND'
        import re
        parts = re.split(r'\s+and\s+|\s+AND\s+', sql_clause)
        
        for part in parts:
            part = part.strip()
            
            # Parse different operators
            # IMPORTANT: Check longer operators first (>= before >, <= before <)
            if '>=' in part:
                # Greater or equal: similarity >= 0.3
                field, value = self._parse_comparison(part, '>=')
                min_key = f'{field}_min'
                conditions[min_key] = value
                
            elif '<=' in part:
                # Less or equal: similarity <= 0.9
                field, value = self._parse_comparison(part, '<=')
                max_key = f'{field}_max'
                conditions[max_key] = value
                
            elif '==' in part or '=' in part:
                # Equality: NSFW == 'UNLIKELY'
                field, value = self._parse_equality(part)
                conditions[field] = value
                
            elif '>' in part:
                # Greater than: original_width > 1024
                field, value = self._parse_comparison(part, '>')
                min_key = f'{field}_min'
                conditions[min_key] = value + 0.0001  # Strict inequality
                
            elif '<' in part:
                # Less than: similarity < 0.5
                field, value = self._parse_comparison(part, '<')
                max_key = f'{field}_max'
                conditions[max_key] = value - 0.0001
        
        return conditions
    
    def _parse_equality(self, expr):
        """Parse equality expression: NSFW == 'UNLIKELY'"""
        if '==' in expr:
            left, right = expr.split('==')
        else:
            left, right = expr.split('=')
        
        field = left.strip()
        value = right.strip().strip('"').strip("'")
        
        # Map SQL field names to IndexManager field names
        field_map = {
            'NSFW': 'nsfw',
            'LICENSE': 'license',
            'similarity': 'similarity',
            'original_width': 'width',
            'original_height': 'height'
        }
        
        return field_map.get(field, field.lower()), value
    
    def _parse_comparison(self, expr, operator):
        """Parse comparison expression: original_width > 1024"""
        left, right = expr.split(operator)
        
        field = left.strip()
        value = float(right.strip())
        
        # Map SQL field names to IndexManager field names
        field_map = {
            'similarity': 'similarity',
            'original_width': 'width',
            'original_height': 'height'
        }
        
        return field_map.get(field, field.lower()), value


# Global instance (similar to duckdb_rel)
index_rel = IndexedMetadataFilter()


# ============================================================================
# Convenience functions (to match duckdb_rel interface)
# ============================================================================

def db_implementation(meta_path):
    """Initialize indexed metadata system"""
    return index_rel.db_implementation(meta_path)


def run_query(res, sql_where_clause):
    """Run query using indexed metadata"""
    filter_obj = res.get('filter_obj')
    if filter_obj is None:
        filter_obj = index_rel
    return filter_obj.run_query(res, sql_where_clause)
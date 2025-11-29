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
            
            print(f"Built indices for {mgr.n:,} rows")
            print(f"Global range: {self.file_ranges[-1]['global_start']}-{self.file_ranges[-1]['global_end']}")
        
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
        
        # Separate special conditions (field-vs-field, NULL checks) from regular conditions
        special_conditions = {}
        regular_conditions = {}
        
        for key, value in conditions.items():
            if '_vs_' in key or '_is_null' in key or '_is_not_null' in key:
                special_conditions[key] = value
            else:
                regular_conditions[key] = value
        
        # Query each file and collect results
        all_results = []
        
        for file_range in self.file_ranges:
            mgr = file_range['manager']
            
            # First: Apply regular IndexManager conditions (uses indices)
            if regular_conditions:
                local_indices = mgr.query(**regular_conditions)
            else:
                # No regular conditions, start with all indices
                local_indices = np.arange(mgr.n, dtype=np.int32)
            
            # Second: Apply special filtering (field-vs-field, NULL checks)
            if special_conditions and len(local_indices) > 0:
                local_indices = self._apply_special_filters(
                    mgr, local_indices, special_conditions
                )
            
            # Convert to global indices
            if len(local_indices) > 0:
                global_indices = local_indices + file_range['global_start']
                all_results.extend(global_indices)
        
        result_array = np.array(all_results, dtype=np.int64)
        
        print(f"Query returned {len(result_array)} rows out of {self.total_rows} total rows.")
        
        return result_array
    
    def _apply_special_filters(self, mgr, indices, conditions):
        """
        Apply special filtering: field-vs-field comparisons and NULL checks
        
        Args:
            mgr: IndexManager instance
            indices: Candidate indices from regular query
            conditions: Dict of special conditions
        
        Returns:
            np.array: Filtered indices
        """
        mask = np.ones(len(indices), dtype=bool)
        
        for condition_key, condition_value in conditions.items():
            # Handle IS NULL
            if condition_key.endswith('_is_null'):
                field = condition_key.replace('_is_null', '')
                if field in mgr.arrays:
                    values = mgr.arrays[field][indices]
                    # Check for NaN or None
                    mask &= pd.isna(values)
                else:
                    print(f"Warning: Field '{field}' not found for NULL check")
                continue
            
            # Handle IS NOT NULL
            if condition_key.endswith('_is_not_null'):
                field = condition_key.replace('_is_not_null', '')
                if field in mgr.arrays:
                    values = mgr.arrays[field][indices]
                    # Check for NOT NaN/None
                    mask &= ~pd.isna(values)
                else:
                    print(f"Warning: Field '{field}' not found for NOT NULL check")
                continue
            
            # Handle field-vs-field comparisons
            if '_vs_' in condition_key:
                # Parse condition: 'width_vs_height_gt' → width > height
                parts = condition_key.split('_vs_')
                field1 = parts[0]
                rest = parts[1].split('_')
                
                # Extract field2 and operator
                # e.g., 'height_gt' → field2='height', op='gt'
                operator = rest[-1]
                field2 = '_'.join(rest[:-1])
                
                # Get arrays for both fields
                array1 = mgr.arrays.get(field1)
                array2 = mgr.arrays.get(field2)
                
                if array1 is None or array2 is None:
                    print(f"Warning: Cannot find arrays for '{field1}' or '{field2}'")
                    continue
                
                # Get values for candidate indices
                values1 = array1[indices]
                values2 = array2[indices]
                
                # Apply operator
                if operator == 'gt':
                    mask &= (values1 > values2)
                elif operator == 'gte':
                    mask &= (values1 >= values2)
                elif operator == 'lt':
                    mask &= (values1 < values2)
                elif operator == 'lte':
                    mask &= (values1 <= values2)
                elif operator == 'eq':
                    # For equality, handle both integer and float types
                    # If both are integer types, use exact equality
                    # If either is float, use close comparison (to handle floating point precision)
                    if np.issubdtype(values1.dtype, np.integer) and np.issubdtype(values2.dtype, np.integer):
                        # Integer comparison - exact equality
                        mask &= (values1 == values2)
                    else:
                        # Float comparison - use np.isclose for robustness
                        # This handles floating point precision issues
                        mask &= np.isclose(values1, values2, rtol=1e-9, atol=1e-9)
                else:
                    print(f"Warning: Unknown operator '{operator}'")
        
        return indices[mask]
    
    def _parse_sql_to_conditions(self, sql_clause):
        """
        Parse SQL WHERE clause into IndexManager query conditions
        
        Supports:
        - field == value / field = value
        - field != value
        - field > value, field >= value, field < value, field <= value
        - field > other_field (field-vs-field)
        - field = other_field (field-vs-field equality)
        - field is NULL
        - field is not NULL
        
        Examples:
            "NSFW == 'UNLIKELY'" → {'nsfw': 'UNLIKELY'}
            "original_width > 1024" → {'width_min': 1024.0001}
            "original_width > original_height" → {'width_vs_height_gt': None}
            "license is NULL" → {'license_is_null': True}
            "license is not NULL" → {'license_is_not_null': True}
            "nsfw = \"NSFW\"" → {'nsfw': 'NSFW'}
        """
        conditions = {}
        
        # Clean up the clause
        sql_clause = sql_clause.strip()
        
        # Field name mapping
        field_map = {
            'NSFW': 'nsfw',
            'nsfw': 'nsfw',
            'LICENSE': 'license',
            'license': 'license',
            'similarity': 'similarity',
            'original_width': 'width',
            'original_height': 'height'
        }
        
        # Split by 'and' or 'AND'
        import re
        parts = re.split(r'\s+and\s+|\s+AND\s+', sql_clause)
        
        for part in parts:
            part = part.strip()
            part_lower = part.lower()
            
            # ========== Handle IS NULL ==========
            if ' is null' in part_lower:
                field = part_lower.split(' is null')[0].strip()
                field_clean = field_map.get(field, field)
                conditions[f'{field_clean}_is_null'] = True
                print(f"  → Parsed: {field_clean} IS NULL")
                continue
            
            # ========== Handle IS NOT NULL ==========
            if ' is not null' in part_lower:
                field = part_lower.split(' is not null')[0].strip()
                field_clean = field_map.get(field, field)
                conditions[f'{field_clean}_is_not_null'] = True
                print(f"  → Parsed: {field_clean} IS NOT NULL")
                continue
            
            # ========== Handle comparison operators ==========
            # Check longer operators first (>= before >, <= before <)
            
            # != (not equal)
            if '!=' in part:
                field, value = self._parse_equality(part, '!=')
                conditions[f'{field}_ne'] = value
                print(f"  → Parsed: {field} != {value}")
                continue
            
            # >= (greater or equal)
            if '>=' in part:
                result = self._parse_comparison(part, '>=')
                if len(result) == 3 and result[2] == 'field':
                    # Field vs Field
                    field, other_field, _ = result
                    conditions[f'{field}_vs_{other_field}_gte'] = None
                    print(f"  → Parsed: {field} >= {other_field} (field vs field)")
                else:
                    # Field vs Value
                    field, value, _ = result
                    conditions[f'{field}_min'] = value
                    print(f"  → Parsed: {field} >= {value}")
                continue
                
            # <= (less or equal)
            if '<=' in part:
                result = self._parse_comparison(part, '<=')
                if len(result) == 3 and result[2] == 'field':
                    # Field vs Field
                    field, other_field, _ = result
                    conditions[f'{field}_vs_{other_field}_lte'] = None
                    print(f"  → Parsed: {field} <= {other_field} (field vs field)")
                else:
                    # Field vs Value
                    field, value, _ = result
                    conditions[f'{field}_max'] = value
                    print(f"  → Parsed: {field} <= {value}")
                continue
            
            # == (equality - check before single =)
            if '==' in part:
                field, value = self._parse_equality(part, '==')
                conditions[field] = value
                print(f"  → Parsed: {field} == '{value}'")
                continue
            
            # > (greater than)
            if '>' in part:
                result = self._parse_comparison(part, '>')
                if len(result) == 3 and result[2] == 'field':
                    # Field vs Field
                    field, other_field, _ = result
                    conditions[f'{field}_vs_{other_field}_gt'] = None
                    print(f"  → Parsed: {field} > {other_field} (field vs field)")
                else:
                    # Field vs Value
                    field, value, _ = result
                    conditions[f'{field}_min'] = value + 0.0001  # Strict inequality
                    print(f"  → Parsed: {field} > {value}")
                continue
                
            # < (less than)
            if '<' in part:
                result = self._parse_comparison(part, '<')
                if len(result) == 3 and result[2] == 'field':
                    # Field vs Field
                    field, other_field, _ = result
                    conditions[f'{field}_vs_{other_field}_lt'] = None
                    print(f"  → Parsed: {field} < {other_field} (field vs field)")
                else:
                    # Field vs Value
                    field, value, _ = result
                    conditions[f'{field}_max'] = value - 0.0001  # Strict inequality
                    print(f"  → Parsed: {field} < {value}")
                continue
            
            # = (single equal - can be field vs field or field vs value)
            if '=' in part:
                left, right = part.split('=', 1)  # Split only on first =
                left = left.strip()
                right = right.strip()
                
                # Map field name
                field = field_map.get(left, left.lower())
                
                # Check quote type
                has_single_quotes = right.startswith("'") and right.endswith("'")
                has_double_quotes = right.startswith('"') and right.endswith('"')
                
                # Remove quotes
                right_clean = right.strip('"').strip("'")
                
                # In SQL:
                # - Single quotes ('value') = string literal
                # - Double quotes ("identifier") = field/table name
                # We need to match DuckDB's behavior
                
                if has_single_quotes:
                    # Single quotes: definitely a string value
                    value = right_clean
                    conditions[field] = value
                    print(f"  → Parsed: {field} = '{value}'")
                elif has_double_quotes:
                    # Double quotes: identifier (field name) in SQL
                    # Check if it maps to a known field
                    if right_clean in field_map:
                        right_field = field_map[right_clean]
                        # Check if it's self-comparison (e.g., nsfw = "NSFW" where "NSFW" maps to nsfw)
                        if right_field == field:
                            # Self-comparison: always true (except for NULL)
                            # We can skip this condition or treat it as "is not NULL"
                            print(f"  → Parsed: {field} = {field} (self-comparison, always true, skipping)")
                            # Don't add any condition - equivalent to always true
                        else:
                            # Different fields
                            conditions[f'{field}_vs_{right_field}_eq'] = None
                            print(f"  → Parsed: {field} = {right_field} (field vs field)")
                    else:
                        # Unknown identifier, treat as field name anyway
                        # This matches DuckDB's lenient behavior
                        right_field = right_clean.lower()
                        if right_field == field:
                            print(f"  → Parsed: {field} = {field} (self-comparison, always true, skipping)")
                        else:
                            conditions[f'{field}_vs_{right_field}_eq'] = None
                            print(f"  → Parsed: {field} = {right_field} (field vs field)")
                elif right_clean in field_map:
                    # No quotes, but is a known field name
                    right_field = field_map[right_clean]
                    if right_field == field:
                        print(f"  → Parsed: {field} = {field} (self-comparison, always true, skipping)")
                    else:
                        conditions[f'{field}_vs_{right_field}_eq'] = None
                        print(f"  → Parsed: {field} = {right_field} (field vs field)")
                else:
                    # No quotes, not a known field: treat as unquoted value
                    value = right_clean
                    conditions[field] = value
                    print(f"  → Parsed: {field} = '{value}'")
                continue
            
            # If we get here, couldn't parse this part
            print(f"Warning: Could not parse: '{part}'")
        
        return conditions
    
    def _parse_equality(self, expr, operator='=='):
        """
        Parse equality expression
        
        Args:
            expr: Expression like "NSFW == 'UNLIKELY'" or "field != 'value'"
            operator: '==' or '!='
        
        Returns:
            tuple: (field_name, value)
        """
        left, right = expr.split(operator, 1)
        
        field = left.strip()
        value = right.strip().strip('"').strip("'")
        
        # Map SQL field names to IndexManager field names
        field_map = {
            'NSFW': 'nsfw',
            'nsfw': 'nsfw',
            'LICENSE': 'license',
            'license': 'license',
            'similarity': 'similarity',
            'original_width': 'width',
            'original_height': 'height'
        }
        
        return field_map.get(field, field.lower()), value
    
    def _parse_comparison(self, expr, operator):
        """
        Parse comparison expression
        Supports both:
        - Field vs Value: original_width > 1024
        - Field vs Field: original_width > original_height
        
        Args:
            expr: Expression like "original_width > 1024"
            operator: '>', '>=', '<', or '<='
        
        Returns:
            tuple: (field, value/other_field, 'value'/'field')
        """
        left, right = expr.split(operator, 1)
        field = left.strip()
        right_str = right.strip()
        
        # Map SQL field names to IndexManager field names
        field_map = {
            'similarity': 'similarity',
            'original_width': 'width',
            'original_height': 'height',
            'NSFW': 'nsfw',
            'LICENSE': 'license'
        }
        
        # Check if right side is a field name or a value
        # If it's a known field name, treat as field comparison
        if right_str in field_map:
            # Field vs Field comparison
            right_field = field_map[right_str]
            return field_map.get(field, field.lower()), right_field, 'field'
        else:
            # Try to parse as numeric value
            try:
                value = float(right_str)
                return field_map.get(field, field.lower()), value, 'value'
            except ValueError:
                # If not numeric, might be a field name not in map
                # Check if it looks like a field (contains letters/underscores)
                if right_str.replace('_', '').isalpha():
                    # Assume it's a field
                    right_field = field_map.get(right_str, right_str.lower())
                    return field_map.get(field, field.lower()), right_field, 'field'
                else:
                    # Give up, return as value
                    return field_map.get(field, field.lower()), right_str, 'value'


# Global instance (similar to duckdb_rel)
index_rel = IndexedMetadataFilter()


# ============================================================================
# Convenience functions (match duckdb_rel interface)
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

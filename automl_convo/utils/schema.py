"""
This module contains functions to infer and build a schema from a pandas DataFrame.
The schema includes data types, unique value counts, and missing value counts for each column.
"""

from typing import Any, Dict
import pandas as pd

def infer_schema_from_df(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    schema = {}
    for col in df.columns:
        series = df[col]
        if pd.api.types.is_numeric_dtype(series):
            col_type = "numeric"
        else:
            col_type = "categorical"
        schema[col] = {
            "type": col_type,
            "unique": series.nunique(dropna=True),
            "missing": int(series.isna().sum()),
        }
    return schema

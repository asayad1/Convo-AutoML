"""
This file depends a feature transformer class which the Feature Engineer agent uses
to augment datasets.
"""

import pandas as pd
from typing import Dict, Any
import re

class FeatureTransformer:
    TRANSFORM_DISPATCH: Dict[str, Any] = {}

    @classmethod
    def get_dispatch(cls) -> Dict[str, Any]:
        """Return the mapping of transform names to handlers."""
        return cls.TRANSFORM_DISPATCH

    @staticmethod
    def apply_add_missing_indicator(df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        src = params["source_column"]
        tgt = params["target_column"]
        df[tgt] = df[src].isna().astype(int)
        return df

    @staticmethod
    def apply_numeric_sum(df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        cols = params["source_columns"]
        tgt = params["target_column"]
        bias = params.get("bias", 0.0)
        df[tgt] = df[cols].sum(axis=1) + bias
        return df

    @staticmethod
    def apply_numeric_ratio(df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        num = params["numerator"]
        den = params["denominator"]
        tgt = params["target_column"]
        eps = params.get("eps", 1e-8)
        df[tgt] = df[num] / (df[den] + eps)
        return df

    @staticmethod
    def apply_text_regex_extract(df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        src = params["source_column"]
        tgt = params["target_column"]
        pattern = params["pattern"]
        group = params.get("group", 1)
        missing_placeholder = params.get("missing_placeholder", "Unknown")

        def extract(val):
            if pd.isna(val):
                return missing_placeholder
            m = re.match(pattern, str(val))
            if not m:
                return missing_placeholder
            try:
                return m.group(group).strip()
            except IndexError:
                return missing_placeholder

        df[tgt] = df[src].apply(extract)
        return df

    @staticmethod
    def apply_text_prefix(df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        src = params["source_column"]
        tgt = params["target_column"]
        n_chars = params.get("n_chars", 1)
        missing_placeholder = params.get("missing_placeholder", "Unknown")

        def prefix(val):
            if pd.isna(val):
                return missing_placeholder
            s = str(val).strip()
            if not s:
                return missing_placeholder
            return s[:n_chars]

        df[tgt] = df[src].apply(prefix)
        return df


# Fill dispatch
FeatureTransformer.TRANSFORM_DISPATCH = {
    "add_missing_indicator": FeatureTransformer.apply_add_missing_indicator,
    "numeric_sum": FeatureTransformer.apply_numeric_sum,
    "numeric_ratio": FeatureTransformer.apply_numeric_ratio,
    "text_regex_extract": FeatureTransformer.apply_text_regex_extract,
    "text_prefix": FeatureTransformer.apply_text_prefix,
}

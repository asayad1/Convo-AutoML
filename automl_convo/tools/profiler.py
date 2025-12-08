"""
This file defines the data profiling node.
"""

from utils.logger import Logger
from utils.schema import infer_schema_from_df
from states.auto_ml_state import AutoMLState

def profile_node(state: AutoMLState) -> AutoMLState:
    logger = Logger()

    logger.info("[PROFILE NODE] Starting data profiling...", style='cyan')

    # Use current dataset
    if state.df_current is None:
        raise ValueError("state.df_current is None in profile_node - expected it to be set.")

    df = state.df_current
    state.n_rows, state.n_cols = df.shape

    logger.info(f"[PROFILE NODE] Loaded dataset with {state.n_rows} rows, {state.n_cols} cols", style='cyan')

    # Rebuild schema from current df that includes engineered features
    state.schema = infer_schema_from_df(df)

    # Log first few columns
    preview_cols = list(state.schema.keys())[:8]
    summary_str = ", ".join(
        f"{col}({state.schema[col]['type']})" for col in preview_cols
    )
    logger.info(f"[PROFILE NODE] Columns summary: {summary_str}...", style='cyan')

    return state

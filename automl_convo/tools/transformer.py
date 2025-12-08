"""
This file defines the data transformation tool.
"""

from utils.logger import Logger
from states.auto_ml_state import AutoMLState
from utils.feature_transformer import FeatureTransformer
import os
import uuid

def apply_transformations_node(state: AutoMLState) -> AutoMLState:
    logger = Logger()

    plan = state.feature_engineer_plan or {"apply": False, "transformations": []}
    apply_flag = plan.get("apply", False)
    transforms = plan.get("transformations", [])

    if not apply_flag or not transforms:
        logger.info("[FEATURE ENGINEER] No transformations to apply.", style="blue")
        state.last_transforms_applied = []
        
        # Ensure df_current is at least df_raw on first iteration
        if state.df_current is None and state.df_raw is not None:
            state.df_current = state.df_raw.copy()
        return state

    # Get the dispatch table from your FeatureTransformer class
    dispatch = FeatureTransformer.get_dispatch()

    df = state.df_current.copy()

    # Collect messages for a nice box log
    box_lines = []
    applied = []

    for t in transforms:
        name = t["name"]
        desc = t.get("description", "")
        params = t.get("params", {})

        fn = dispatch.get(name)
        if fn is None:
            box_lines.append(f"[yellow]Skipped unknown transform:[/yellow] {name}")
            continue

        box_lines.append(f"[cyan]{name}[/cyan] - {desc}")
        df = fn(df, params)
        applied.append({"name": name, "description": desc, "params": params})

    # Print everything in a rich box
    logger.box(
        "FEATURE ENGINEERING - Applying Transformations",
        "\n".join(box_lines),
        style="blue",
    )

    # Update current dataframe
    state.df_current = df

    # Ensure temp dir exists
    os.makedirs(state.temp_dir, exist_ok=True)
    iter_id = state.iteration
    fname = f"iter{iter_id}_{uuid.uuid4().hex[:8]}.csv"
    out_path = os.path.join(state.temp_dir, fname)
    df.to_csv(out_path, index=False)

    state.current_dataset_csv = out_path
    state.datasets_history.append(out_path)
    state.last_transforms_applied = applied

    logger.info(f"[FEATURE ENGINEER] Saved augmented dataset to: {out_path}", style="blue")
    return state
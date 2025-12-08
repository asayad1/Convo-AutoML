"""
This file defines the feature engineer agent.
"""

from states.auto_ml_state import AutoMLState
from utils.logger import Logger
from llm import LLM
import json
import re 

def feature_engineer_node(state: AutoMLState, llm: LLM) -> AutoMLState:
    logger = Logger()

    logger.info("[FEATURE ENGINEER NODE] Proposing new features...", style='blue')

    # Build compact schema summary
    schema_str = "\n".join(
        f"- {col}: type={meta['type']}, unique={meta['unique']}, missing={meta['missing']}"
        for col, meta in state.schema.items()
    )

    # Provide last iteration context to LLM
    last_result_summary = ""
    if state.history:
        last = state.history[-1]
        best_model = max(
            last["model_results"].items(),
            key=lambda kv: kv[1]["mean_score"],
        )
        last_feats = ", ".join(last["used_features"][:10])
        last_transforms = ", ".join(
            t["name"] for t in last["transforms_applied"]
        )
        
        last_result_summary = f"""
Last iteration:
- Best model: {best_model[0]} (mean_score={best_model[1]['mean_score']:.4f})
- Used features (first 10): {last_feats}
- Transforms applied: {last_transforms or 'none'}
""".strip()

    system_prompt = """
You are a feature engineering agent operating on a general tabular dataset.

Inputs:
- Dataset schema (column names, types, unique counts, missing counts).
- Task type (classification or regression).
- Target column.
- Optional context from previous iterations.

You may propose up to 3 transformations in a single step. Transformations must be applied to work 
towards answering the user's question (if given the context). You MUST ONLY USE columns that exist in the 
PROVIDED schema.

Available transformation types (name field):
- "add_missing_indicator": add a binary column indicating missingness of another column.
    params: { "source_column": str, "target_column": str }
- "numeric_sum": sum multiple numeric columns (optionally + bias).
    params: { "source_columns": [str, ...], "target_column": str, "bias": float (optional) }
- "numeric_ratio": ratio between two numeric columns.
    params: { "numerator": str, "denominator": str, "target_column": str, "eps": float (optional) }
- "text_regex_extract": extract text via regex into a new categorical column.
    params: { "source_column": str, "target_column": str, "pattern": str, "group": int, "missing_placeholder": str }
- "text_prefix": extract the first N characters of a text column into a new categorical column.
    params: { "source_column": str, "target_column": str, "n_chars": int, "missing_placeholder": str }

Respond ONLY with JSON of the form:
{
  "apply": true or false,
  "rationale": "short internal rationale",
  "transformations": [
    {
      "name": "<one of the names above>",
      "description": "short human-readable description",
      "params": { ... }
    },
    ...
  ]
}
"""

    human_prompt = f"""
Task type: {state.task_type}
Target column: {state.target_column}

Dataset schema:
{schema_str}

{last_result_summary if last_result_summary else "No previous iterations."}

Design generic transformations that could help this task, without assuming a specific domain.
"""

    content = llm.invoke(system_prompt, human_prompt)

    try:
        obj = json.loads(content)
    except Exception:
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if not match:
            raise ValueError("Could not parse feature engineer response as JSON.")
        obj = json.loads(match.group(0))

    state.feature_engineer_plan = obj
    return state

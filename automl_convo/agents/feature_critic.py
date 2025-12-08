"""
This file defines the feature critic agent.
"""

from utils.logger import Logger
from states.auto_ml_state import AutoMLState
from llm import LLM
import re 
import json 


def feature_critic_node(state: AutoMLState, llm: LLM) -> AutoMLState:
    logger = Logger()

    logger.info("[FEATURE CRITIC NODE] Critiquing features and model performance...", style='magenta')

    last = state.history[-1]
    schema_str = "\n".join(
        f"- {col}: type={meta['type']}, unique={meta['unique']}, missing={meta['missing']}"
        for col, meta in state.schema.items()
    )

    results_str = "\n".join(
        f"- {name}: mean={res['mean_score']:.4f}, std={res['std']:.4f}"
        for name, res in last["model_results"].items()
    )
    transforms_str = ", ".join(t["name"] for t in last["transforms_applied"]) or "none"

    system_prompt = """
You are a feature critic agent for a general tabular dataset.

You are given:
- Task type and target column.
- Dataset schema.
- The most recent iteration's model results (accuracy/R^2 means & stds).
- The list of features used (column names only).
- The transformations that were just applied.

Your job:
- Decide whether another round of feature engineering is likely to help.
- If yes, propose up to 3 NEW transformations (using the same generic transform types as the feature engineer agent).

Respond ONLY with JSON:
{
  "apply": true or false,
  "rationale": "short internal rationale",
  "transformations": [
    {
      "name": "<transform_name>",
      "description": "short description",
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

Last iteration:
- Features used (first 15): {", ".join(last["used_features"][:15])}
- Model results:
{results_str}
- Transforms applied: {transforms_str}

You are at iteration {state.iteration} out of max {state.max_iterations}.
If we are already at the maximum iteration, you MUST set apply=false.
Otherwise, only propose additional feature engineering if it is likely to improve performance.
"""

    content = llm.invoke(system_prompt, human_prompt)

    try:
        obj = json.loads(content)
    except Exception:
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if not match:
            raise ValueError("Could not parse feature critic response as JSON.")
        obj = json.loads(match.group(0))

    # If we've hit max iterations, force apply=false
    if state.iteration >= state.max_iterations:
        obj["apply"] = False
        obj["rationale"] = obj.get("rationale", "") + " (Overridden: max iterations reached.)"
        obj["transformations"] = []

    state.feature_critic_plan = obj
    # Build a readable multiline summary
    plan_str = (
        f"apply: {obj.get('apply')}\n"
        f"rationale: {obj.get('rationale')}\n"
    )

    # If transformations exist, list them
    transforms = obj.get("transformations", [])
    if transforms:
        plan_str += "\ntransformations:\n"
        for t in transforms:
            plan_str += f"  - {t.get('name')}: {t.get('description')}\n"

    # Log in a panel
    logger.box(
        "FEATURE CRITIC PLAN",
        plan_str,
        style="magenta",
    )

    return state

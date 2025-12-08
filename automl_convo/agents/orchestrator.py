"""
This file implements the orchestrator agent, which determines the type of task, the target,
and the whether we want to use PCA.
"""

from utils.logger import Logger
from states.auto_ml_state import AutoMLState
from llm import LLM
import json
import re 

def orchestrator_node(state: AutoMLState, llm: LLM, question: str) -> AutoMLState:
    logger = Logger()

    logger.info(
        "[ORCHESTRATOR NODE] Deciding target, task type, and PCA usage based on question + schema...",
        style="magenta",
    )

    schema_str = "\n".join(
        f"- {col}: type={meta['type']}, unique={meta['unique']}, missing={meta['missing']}"
        for col, meta in state.schema.items()
    )

    system_prompt = """
You are a data scientist orchestrator for a general tabular dataset tool.

You are given:
- A user question.
- A dataset schema (column names, inferred types, unique counts, missing counts).

Your jobs:
1. Decide which column should be the modeling target.
2. Decide whether the task is "classification" or "regression".
3. Decide whether we should apply PCA or not.
   - Use PCA (use_pca=true) when:
     - The dataset has many features or high-cardinality one-hot encodings,
     - The user cares primarily about PREDICTION quality, and
     - They do NOT explicitly ask for interpretability, feature importance, or coefficients.
   - Avoid PCA (use_pca=false) when:
     - The user asks about feature importance, effect sizes, coefficients, interpretations, or "which features matter" in a detailed way.
     - The number of features is modest and interpretability is more important than squeezing out a tiny bit of performance.
4. If you choose PCA, pick a reasonable small integer for "pca_components" (e.g., 5-50 depending on feature count). If you do not want PCA, set pca_components = null.

Respond ONLY with a JSON object with keys:
- "target_column": string
- "task_type": "classification" or "regression"
- "use_pca": boolean
- "pca_components": integer or null
- "rationale": short string
"""

    human_prompt = f"""
User question:
{question}

Dataset schema:
{schema_str}
"""

    logger.info("[ORCHESTRATOR NODE] Sending prompt to LLM...", style="magenta")
    resp_content = llm.invoke(system_prompt, human_prompt)

    try:
        obj = json.loads(resp_content)
    except Exception:
        # Try to extract JSON substring
        match = re.search(r"\{.*\}", resp_content, re.DOTALL)
        if not match:
            raise ValueError("Could not parse LLM orchestrator response as JSON.")
        obj = json.loads(match.group(0))

    target_column = obj["target_column"]
    task_type = obj["task_type"]
    rationale = obj.get("rationale", "")
    use_pca = bool(obj.get("use_pca", True))
    pca_components = obj.get("pca_components")

    if pca_components is None:
        # A small default if PCA is used
        if use_pca:
            pca_components = min(20, max(5, state.n_cols - 1))
        else:
            # we won't use it anyway
            pca_components = 0

    logger.info(f"[ORCHESTRATOR NODE] Chosen target_column: {target_column}", style="magenta")
    logger.info(f"[ORCHESTRATOR NODE] Chosen task_type: {task_type}", style="magenta")
    logger.info(
        f"[ORCHESTRATOR NODE] use_pca: {use_pca}, pca_components: {pca_components}",
        style="magenta",
    )

    state.target_column = target_column
    state.task_type = task_type
    state.orchestration_rationale = rationale
    state.use_pca = use_pca
    state.pca_components = int(pca_components) if pca_components else 0

    return state

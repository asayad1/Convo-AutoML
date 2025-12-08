"""
This file implements the user facing analyst agent.
"""

from utils.logger import Logger
from states.auto_ml_state import AutoMLState
from llm import LLM

def analysis_node(state: AutoMLState, llm: LLM, question: str) -> AutoMLState:
    logger = Logger()

    logger.info("[ANALYSIS NODE] Asking LLM for final user-facing analysis...", style="cyan")

    # Build a compact history string
    hist_lines = []
    for h in state.history:
        models_str = ", ".join(
            f"{mname}: mean={res['mean_score']:.4f}, std={res['std']:.4f}"
            for mname, res in h["model_results"].items()
        )
        transforms_str = ", ".join(t["name"] for t in h["transforms_applied"]) or "none"

        feat_metrics = h.get("feature_metrics")
        if feat_metrics and feat_metrics.get("feature_importances"):
            top_feats = feat_metrics["feature_importances"][:10]
            feat_str = ", ".join(
                f"{fi['feature']} (norm_importance={fi['importance_norm']:.3f})"
                for fi in top_feats
            )
        else:
            feat_str = "none"

        hist_lines.append(
            f"Iteration {h['iteration']}:\n"
            f"  - dataset_csv: {h['dataset_csv']}\n"
            f"  - used_features (first 10): {', '.join(h['used_features'][:10])}\n"
            f"  - transforms_applied: {transforms_str}\n"
            f"  - model_results: {models_str}\n"
            f"  - top_feature_importances: {feat_str}\n"
        )

    history_str = "\n".join(hist_lines)

    system_prompt = """
You are an experienced data scientist.

You are given:
- A user question about a general tabular dataset.
- The chosen target column and task type.
- A history of several AutoML iterations where:
  - Each iteration has: dataset path, list of features used, transformations applied, model results (mean scores), and optional feature-level importance metrics for the best model.

Your job:
1. Synthesize what the models learned that is relevant to the user's question.
2. Describe which features (or feature groups) ended up being most important or most useful, using the provided feature importance metrics when available.
3. Briefly compare the models' performances and what that implies about the underlying signal.
4. Give concrete, plain-language insights about the dataset, not generic AutoML commentary.
5. Do NOT invent quantitative effects or subgroup values that are not implied by the metrics; if the exact size of an effect is unclear, say so qualitatively.
6. You must not claim feature importances for models that did not have feature-importance metrics provided. If feature importances are only given for one model, clearly say so and attribute them only to that model.

Output:
- A clear, concise Markdown answer for the user.
- Do not describe your own reasoning process or say things like "let's craft the answer".
"""

    human_prompt = f"""
User question:
{question}

Target column: {state.target_column}
Task type: {state.task_type}

Iteration history:
{history_str}
"""

    raw = llm.invoke(system_prompt, human_prompt)

    planning_markers = [
        "Let's craft answer",
        "Let's craft the answer",
        "We need to synthesize",
        "We need to",
        "Let's outline",
    ]
    if any(marker in raw for marker in planning_markers):
        logger.info(
            "[ANALYSIS NODE] Detected planning-style output; running second pass to get final Markdown answer...",
            style="yellow",
        )

        system_prompt_2 = """
You are an experienced data scientist.

You are given some internal notes about how to answer a question.
Your task now is:

- Use those notes as background.
- Produce only the final, polished answer in Markdown.
- Do NOT mention the notes, your reasoning steps, or that you are crafting an answer.
- Just write the final answer as if you were talking directly to the user.
"""

        human_prompt_2 = f"""
Internal notes:
{raw}

Now write the final Markdown answer to the original question.
"""

        content = llm.invoke(system_prompt_2, human_prompt_2)
    else:
        content = raw

    state.final_answer = content
    logger.box_md("RESPONSE TO USER", state.final_answer, style="green")
    return state

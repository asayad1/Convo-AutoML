"""
This file implements the model results explainer agent. It uses the existing AutoMLState
to interpret and answer the users question, and avoids recomputation.
"""

from states.auto_ml_state import AutoMLState
from llm import LLM

def model_results_explainer(question: str, llm: LLM, state: AutoMLState) -> str:
    """
    Use the existing AutoMLState (no new training) to answer the user's question.
    This is similar to analysis_node but does NOT modify state or run any new ML.
    """
    if state is None or not state.history:
        return "I don't have any previous modeling results yet; I would need to run a new analysis first."

    # Build history summary
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
            f"  - top_feature_importances_for_best_model: {feat_str}\n"
        )

    history_str = "\n".join(hist_lines)

    system_prompt = """
You are an experienced data scientist. You are given:
- A user question about a tabular dataset.
- A fixed set of previously computed AutoML results (no new training allowed).
  Each iteration summary includes model scores and OPTIONAL feature importances
  for the best model.

Your job:
1. Answer the user's question only using the information from the existing results.
2. Do NOT assume any new modeling or feature engineering has been run.
3. Use feature importances and model scores when available.
4. If you cannot fully answer the question from the existing results, say so explicitly
   and explain what additional analysis would be needed.

Constraints:
- Do NOT invent new quantitative effects or metrics that are not implied by the history.
- Do NOT claim feature importances for models that do not have such metrics provided.
- Output a clear, concise answer in Markdown.
"""

    human_prompt = f"""
User question:
{question}

Existing AutoML results:
Target column: {state.target_column}
Task type: {state.task_type}

Iteration history:
{history_str}
"""

    content = llm.invoke(system_prompt, human_prompt)
    return content

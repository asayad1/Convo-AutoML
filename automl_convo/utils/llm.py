"""
This file defines various utilities from the LLM.
"""

from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from states.auto_ml_state import AutoMLState
from typing import Dict, Any, List
import numpy as np


def summarize_automl_state_for_llm(state: AutoMLState) -> str:
    """
    Create a compact, text summary of what has been done so far:
    - target, task_type
    - model results across iterations
    - top feature importances
    """
    if state is None or not state.history:
        return "No previous AutoML runs have been executed."

    lines = []
    lines.append(f"Target column: {state.target_column}")
    lines.append(f"Task type: {state.task_type}")
    lines.append("Iterations summary:")

    for h in state.history:
        models_str = ", ".join(
            f"{name}: mean={res['mean_score']:.4f}, std={res['std']:.4f}"
            for name, res in h["model_results"].items()
        )
        fm = h.get("feature_metrics")
        if fm and fm.get("feature_importances"):
            top_feats = fm["feature_importances"][:8]
            feats_str = ", ".join(
                f"{fi['feature']} (norm_importance={fi['importance_norm']:.3f})"
                for fi in top_feats
            )
        else:
            feats_str = "none"

        lines.append(
            f"- Iteration {h['iteration']}: "
            f"models = [{models_str}], "
            f"top_feature_importances_for_best_model = {feats_str}"
        )

    return "\n".join(lines)


def build_model(name: str, params: Dict[str, Any]):
    """
    Builds the correct model depending on the models chosen.
    """
    # Classification models
    if name == "logistic_regression":
        return LogisticRegression(**params)
    if name == "decision_tree_clf":
        return DecisionTreeClassifier(**params)
    if name == "mlp_classifier":
        return MLPClassifier(**params)

    # Regression models
    if name == "linear_regression":
        return LinearRegression(**params)
    if name == "decision_tree_reg":
        return DecisionTreeRegressor(**params)
    if name == "mlp_regressor":
        return MLPRegressor(**params)

    raise ValueError(f"Unknown model name '{name}'")


def compute_feature_importances(model, feature_names: List[str]) -> List[Dict[str, Any]]:
    """
    Given a fitted sklearn model and a list of feature names, return
    a list of {feature, importance, importance_norm} sorted by importance.
    """

    importances = None

    # Linear models: absolute coefficients
    if isinstance(model, (LinearRegression, LogisticRegression)):
        coefs = model.coef_
        if hasattr(coefs, "ndim") and coefs.ndim > 1:
            coefs = coefs.mean(axis=0)
        importances = np.abs(coefs)

    # Tree-based models: feature_importances_
    elif isinstance(model, (DecisionTreeRegressor, DecisionTreeClassifier)) or hasattr(model, "feature_importances_"):
        importances = getattr(model, "feature_importances_", None)

    if importances is None:
        return []

    importances = np.asarray(importances)
    n = min(len(feature_names), len(importances))
    pairs = [
        {
            "feature": feature_names[i],
            "importance": float(importances[i]),
        }
        for i in range(n)
    ]

    total = sum(p["importance"] for p in pairs) or 1.0
    for p in pairs:
        p["importance_norm"] = p["importance"] / total

    pairs.sort(key=lambda d: d["importance"], reverse=True)
    return pairs

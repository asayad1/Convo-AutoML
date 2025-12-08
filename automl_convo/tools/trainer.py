"""
This file defines the model trainer tool node.
"""

from states.auto_ml_state import AutoMLState
from typing import Dict, Any
from utils.logger import Logger
from utils.llm import build_model, compute_feature_importances
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold


def train_node(state: AutoMLState) -> AutoMLState:
    logger = Logger()

    logger.info("[TRAIN NODE] Training models and evaluating performance...", style="green")
    X = state.X_processed
    y = state.y

    if X is None or y is None:
        raise ValueError("X_processed or y is None in train_node - check clean_node and target_column.")

    results: Dict[str, Dict[str, Any]] = {}

    # Choose CV strategy + scoring based on task type
    if state.task_type == "classification":
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        scoring = "accuracy"
    elif state.task_type == "regression":
        cv = KFold(n_splits=3, shuffle=True, random_state=42)
        scoring = "r2"
    else:
        raise ValueError(f"Unknown task_type '{state.task_type}' in train_node.")

    planned_params = {mname: params for (mname, params) in state.planned_models}

    for mname, params in state.planned_models:
        model = build_model(mname, params)
        scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)

        results[mname] = {
            "mean_score": float(scores.mean()),
            "std": float(scores.std()),
            "scores": scores.tolist(),
            "metric": scoring,
        }

    state.model_results = results

    # Build training summary for log output
    summary_lines = []

    summary_lines.append("[bold green]Model results summary:[/bold green]")
    for mname, res in results.items():
        summary_lines.append(
            f"- {mname}: mean_{res['metric']}={res['mean_score']:.4f}, "
            f"std={res['std']:.4f}"
        )

    feature_metrics = None

    # Compute feature-level metrics for the best model when PCA is disabled
    if not state.use_pca:
        if state.used_features is not None:
            best_name, best_res = max(results.items(), key=lambda kv: kv[1]["mean_score"])
            summary_lines.append(f"\n[bold green]Best model by {scoring}:[/bold green] {best_name}")

            best_params = planned_params[best_name]
            best_model = build_model(best_name, best_params)
            best_model.fit(X, y)

            importances = compute_feature_importances(best_model, state.used_features)

            feature_metrics = {
                "iteration": state.iteration,
                "best_model": best_name,
                "metric": scoring,
                "mean_score": best_res["mean_score"],
                "feature_importances": importances,
            }

            summary_lines.append("\n[bold green]Top 10 feature importances:[/bold green]")
            for fi in importances[:10]:
                summary_lines.append(
                    f"  {fi['feature']}: importance={fi['importance']:.4f}, "
                    f"norm={fi['importance_norm']:.4f}"
                )

            state.feature_metrics_history.append(feature_metrics)
        else:
            summary_lines.append(
                "[yellow]used_features is None; skipping feature metric computation.[/yellow]"
            )
    else:
        summary_lines.append(
            "[yellow]PCA is enabled; feature-level importances are in component space, "
            "skipping per-original-feature metrics.[/yellow]"
        )

    # Turn lines into a single string and show in a Rich box
    box_content = "\n".join(summary_lines)
    logger.box("TRAINING SUMMARY", box_content, style="green")

    # Update iteration history
    iter_record = {
        "iteration": state.iteration,
        "dataset_csv": state.current_dataset_csv,
        "used_features": state.used_features,
        "model_results": results,
        "transforms_applied": state.last_transforms_applied,
        "feature_metrics": feature_metrics,
    }
    state.history.append(iter_record)

    return state

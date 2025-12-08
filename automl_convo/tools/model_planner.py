"""
This file defines the model planner tool node.
"""

from utils.logger import Logger
from states.auto_ml_state import AutoMLState

def model_plan_node(state: AutoMLState) -> AutoMLState:
    logger = Logger()
    logger.info("[MODEL PLAN NODE] Planning which models to train...", style='orange')

    if state.task_type == "classification":
        planned = [
            ("logistic_regression", {"max_iter": 10000}),
            ("decision_tree_clf", {"max_depth": 5}),
            ("mlp_classifier", {"hidden_layer_sizes": (64,), "max_iter": 10000}),
        ]
    elif state.task_type == "regression":
        planned = [
            ("linear_regression", {}),
            ("decision_tree_reg", {"max_depth": 5}),
            ("mlp_regressor", {"hidden_layer_sizes": (64,), "max_iter": 35}),
        ]
    else:
        raise ValueError(f"Unknown task_type '{state.task_type}'. Expected 'classification' or 'regression'.")

    state.planned_models = planned

    logger.info(f"[MODEL PLAN NODE] Task type: {state.task_type}", style='orange')
    logger.info("[MODEL PLAN NODE] Planned models:", style='orange')
    for mname, params in planned:
        logger.info(f"  - {mname} with params {params}", style='orange')

    return state

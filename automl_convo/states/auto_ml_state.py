"""
This module defines the AutoMLState dataclass to maintain the state of the AutoML process.
It includes dataframes, profiling info, preprocessing details, modeling plans,
feature engineering steps, datasets history, and iteration history.
"""

from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np

@dataclass
class AutoMLState:
    # Core data
    df_raw: Optional[pd.DataFrame] = None
    df_current: Optional[pd.DataFrame] = None
    csv_path: Optional[str] = None

    # Profiling and orchestration
    schema: Optional[Dict[str, Dict[str, Any]]] = None
    n_rows: Optional[int] = None
    n_cols: Optional[int] = None
    target_column: Optional[str] = None
    task_type: Optional[str] = None  # this has to be "classification" or "regression"
    orchestration_rationale: Optional[str] = None

    # PCA settings
    use_pca: bool = True
    pca_components: int = 10

    # Iteration control
    iteration: int = 1
    max_iterations: int = 3

    # Preprocessing and modeling
    X_processed: Optional[np.ndarray] = None
    y: Optional[np.ndarray] = None
    used_features: Optional[List[str]] = None
    planned_models: Optional[List[Tuple[str, Dict[str, Any]]]] = None
    model_results: Optional[Dict[str, Dict[str, Any]]] = None
    clean_pipeline: Optional[Pipeline] = None

    # Feature engineering
    feature_engineer_plan: Optional[Dict[str, Any]] = None
    feature_critic_plan: Optional[Dict[str, Any]] = None
    last_transforms_applied: List[Dict[str, Any]] = field(default_factory=list)

    # Datasets history
    temp_dir: str = "augmented_datasets"
    current_dataset_csv: Optional[str] = None
    datasets_history: List[str] = field(default_factory=list)

    # Iteration history for final analysis
    history: List[Dict[str, Any]] = field(default_factory=list)

    # Feature-level metrics across iterations
    feature_metrics_history: List[Dict[str, Any]] = field(default_factory=list)

    # Final user-facing answer
    final_answer: Optional[str] = None

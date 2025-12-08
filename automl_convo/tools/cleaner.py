"""
This file defines the data cleaner tool node.
"""

from states.auto_ml_state import AutoMLState
from utils.logger import Logger
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import pandas as pd 


def clean_node(state: AutoMLState) -> AutoMLState:
    logger = Logger()

    logger.info("[CLEAN NODE] Building preprocessing pipeline and transforming data...", style="green")

    if state.df_current is None:
        raise ValueError("state.df_current is None in clean_node - expected current dataset.")
    df = state.df_current.copy()
    target = state.target_column

    if target is None:
        raise ValueError("Target column not set in state.target_column.")
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in dataframe.")

    X = df.drop(columns=[target])
    y = df[target].values

    # Type look, and fall back to dtype if not in schema
    numeric_features = []
    categorical_features = []

    for c in X.columns:
        col_meta = state.schema.get(c) if state.schema is not None else None
        if col_meta is not None:
            col_type = col_meta.get("type")
        else:
            col_type = "numeric" if pd.api.types.is_numeric_dtype(X[c]) else "categorical"

        if col_type == "numeric":
            numeric_features.append(c)
        else:
            categorical_features.append(c)

    # Log summary contents
    summary_lines = []
    summary_lines.append(f"target_column: {target}")
    summary_lines.append(f"numeric_features: {numeric_features}")
    summary_lines.append(f"categorical_features: {categorical_features}")

    # Numeric pipeline
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    # Categorical pipeline
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    steps = [("preprocessor", preprocessor)]

    use_pca = state.use_pca
    pca_components = state.pca_components

    if use_pca and pca_components and pca_components > 0:
        steps.append(("pca", PCA(n_components=pca_components)))
        summary_lines.append(f"PCA: True (n_components = {pca_components})")
    else:
        summary_lines.append("PCA: False")

    pipeline = Pipeline(steps=steps)
    X_processed = pipeline.fit_transform(X)

    summary_lines.append(f"X_processed shape: {X_processed.shape}")

    # Render all details in one boxed panel
    logger.box(
        "CLEAN SUMMARY",
        "\n".join(summary_lines),
        style="green",
    )

    # Update state
    state.X_processed = X_processed
    state.y = y
    state.used_features = list(X.columns)
    state.clean_pipeline = pipeline

    return state

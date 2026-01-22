"""Cloud training worker for credit risk prediction.

Purpose:
    Executes a streaming ensemble training pipeline designed for Azure Compute.
    Streams CSV input in chunks and trains incremental SGD classifiers with
    Nystroem kernel approximation for non-linear feature mapping.

Architecture:
    - Incremental Learning: Uses `partial_fit` to train on chunks.
    - Ensemble: Configurable number of SGD classifiers with soft voting.
    - Kernel Approximation: Projects features using Nystroem with `--nystroem_components`.

Author: Shane Lee
Licence: MIT
"""

import argparse
import gc
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import sklearn
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# --- Shared Utils Import ---
try:
    # When running in Azure, this file is injected into the root of the src dir
    import scalability_utils as utils
except ImportError:
    # Fallback if running locally in dev mode without packaging
    sys.path.append(".")
    import scalability_utils as utils  # type: ignore


# --- Logging Infrastructure ---

def setup_forensic_logger() -> logging.Logger:
    """Configures a dedicated forensic logger for the Cloud runtime.

    Returns:
        logging.Logger: Configured logger instance.
    """
    os.makedirs("outputs", exist_ok=True)
    return utils.configure_logging("CloudTrainer", "forensic_audit.log", log_to_file=True)


logger: logging.Logger = setup_forensic_logger()


# --- Core Logic ---

def scan_class_weights(
    data_path: str,
    batch_size: int,
    target_col: str
) -> Tuple[Dict[Any, float], np.ndarray]:
    """Scans the dataset stream to determine global class distribution.

    Args:
        data_path: Input URI or path.
        batch_size: Iteration batch size for pre-scan.
        target_col: Label column name.

    Returns:
        Tuple: (Class weights dictionary, Sorted unique classes array).
    """
    logger.info("Scanning dataset stream for class weights...")
    class_counter: Dict[Any, int] = {}
    total_samples: int = 0

    try:
        with pd.read_csv(data_path, header=1, index_col=0, chunksize=batch_size) as reader:
            for chunk in reader:
                if target_col in chunk.columns:
                    counts: Dict[Any, int] = chunk[target_col].value_counts().to_dict()
                    for cls, count in counts.items():
                        class_counter[cls] = class_counter.get(cls, 0) + count
                        total_samples += count

        all_classes: np.ndarray = np.array(sorted(class_counter.keys()))
        weights: Dict[Any, float] = {
            cls: total_samples / (len(all_classes) * count)
            for cls, count in class_counter.items()
        }
        logger.info(f"Detected Classes: {all_classes} | Weights: {weights}")
        return weights, all_classes
    except Exception as e:
        logger.error(f"CRITICAL: Pre-scan failed. Error: {e}")
        sys.exit(1)


def initialize_ensemble(
    n_estimators: int,
    n_components: int,
    class_weights: Dict[Any, float]
) -> Tuple[StandardScaler, Nystroem, List[SGDClassifier]]:
    """Initialises the incremental ensemble components.

    Args:
        n_estimators: Count of independent models.
        n_components: Dimensionality for kernel approximation.
        class_weights: Balanced weight map.

    Returns:
        Tuple: (Scaler, Kernel Mapper, List of Estimators).
    """
    scaler: StandardScaler = StandardScaler()
    feature_map: Nystroem = Nystroem(
        gamma=0.1,
        n_components=n_components,
        random_state=42
    )

    models: List[SGDClassifier] = [
        SGDClassifier(
            loss='log_loss',
            learning_rate='adaptive',
            eta0=0.01,
            penalty='l2',
            class_weight=class_weights,
            random_state=42 + i,
            n_jobs=1  # We parallelise via job submission, not threads here
        ) for i in range(n_estimators)
    ]

    logger.info(f"Initialised Ensemble with {n_estimators} estimators and {n_components} components.")
    return scaler, feature_map, models


def main() -> None:
    """Main execution flow for Cloud Training."""
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Credit Default Incremental Training Worker"
    )
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--test_train_ratio", type=float, default=0.2)
    parser.add_argument("--learning_rate", type=float, default=0.1)
    parser.add_argument("--max_iter", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=50000)
    parser.add_argument("--n_estimators", type=int, default=7)
    parser.add_argument("--nystroem_components", type=int, default=1200)
    parser.add_argument("--features_json", type=str, required=True)
    parser.add_argument("--registered_model_name", type=str, required=True)
    args: argparse.Namespace = parser.parse_args()

    mlflow.start_run()

    try:
        feature_defs: Dict[str, Any] = json.loads(args.features_json)
        target_col: str = feature_defs.get("target", "default payment next month")
    except json.JSONDecodeError as e:
        logger.error(f"CRITICAL: Feature schema serialisation error. Error: {e}")
        sys.exit(1)

    # --- Step 1: Resource Scoping ---
    class_weights, all_classes = scan_class_weights(args.data, args.batch_size, target_col)
    scaler, feature_map, models = initialize_ensemble(
        args.n_estimators, args.nystroem_components, class_weights
    )

    forensics: utils.StreamForensics = utils.StreamForensics(estimated_rows=10_000_000)
    y_test_agg: List[int] = []
    y_prob_agg: List[float] = []

    # --- Step 2: Streaming Incremental Learning ---
    try:
        for epoch in range(args.max_iter):
            logger.info(f"--- Commencing Epoch {epoch + 1}/{args.max_iter} ---")

            with pd.read_csv(args.data, header=1, index_col=0, chunksize=args.batch_size) as reader:
                for chunk in reader:
                    # Forensic audit established in Epoch 0
                    if epoch == 0:
                        chunk = forensics.audit_chunk(chunk)

                    if chunk.empty or target_col not in chunk.columns:
                        continue

                    y_chunk: pd.Series = chunk.pop(target_col)
                    x_train, x_test, y_train, y_test = train_test_split(
                        chunk, y_chunk, test_size=args.test_train_ratio, random_state=42
                    )

                    # Incremental Scaling and Kernel Projection
                    scaler.partial_fit(x_train)
                    x_train_scaled: np.ndarray = scaler.transform(x_train)

                    if epoch == 0 and not hasattr(feature_map, 'components_'):
                        feature_map.fit(x_train_scaled)

                    x_train_mapped: np.ndarray = feature_map.transform(x_train_scaled)

                    # Online update for each ensemble member
                    for model in models:
                        model.partial_fit(x_train_mapped, y_train, classes=all_classes)

                    # Final Epoch Evaluation (Accumulate test set results)
                    if epoch == args.max_iter - 1:
                        x_test_scaled: np.ndarray = scaler.transform(x_test)
                        x_test_mapped: np.ndarray = feature_map.transform(x_test_scaled)

                        # Soft voting aggregation
                        probs: List[np.ndarray] = [m.predict_proba(x_test_mapped)[:, 1] for m in models]
                        y_test_agg.extend(y_test.tolist())
                        y_prob_agg.extend(np.mean(probs, axis=0).tolist())

                    del chunk, x_train, x_test, y_train, y_test
                    gc.collect()

    except Exception as e:
        logger.error(f"CRITICAL: Learning stream interrupted. Error: {e}")
        sys.exit(1)

    forensics.log_summary()

    # --- Step 3: Forensic Metric Derivation ---
    if not y_test_agg:
         logger.error("CRITICAL: No test data accumulated. Validation failed.")
         sys.exit(1)

    y_test_arr: np.ndarray = np.array(y_test_agg)
    y_prob_arr: np.ndarray = np.array(y_prob_agg)
    y_pred_arr: np.ndarray = (y_prob_arr >= 0.5).astype(int)

    metrics: Dict[str, float] = {
        "accuracy": accuracy_score(y_test_arr, y_pred_arr),
        "f1_weighted": f1_score(y_test_arr, y_pred_arr, average='weighted'),
        "precision_weighted": precision_score(y_test_arr, y_pred_arr, average='weighted', zero_division=0),
        "recall_weighted": recall_score(y_test_arr, y_pred_arr, average='weighted', zero_division=0),
        "auc_roc": roc_auc_score(y_test_arr, y_prob_arr)
    }

    logger.info(f"SCORING COMPLETE - AUC: {metrics['auc_roc']:.4f} | Accuracy: {metrics['accuracy']:.4f}")
    mlflow.log_metrics(metrics)

    # --- Step 4: Artifact Serialisation ---
    # We pipeline the scaler and mapper for inference convenience, but keep models separate list
    # for the ensemble logic.
    artifact: Dict[str, Any] = {
        'preprocessor': make_pipeline(scaler, feature_map),
        'ensemble_models': models,
        'metadata': {
            'sklearn': sklearn.__version__,
            'training_mode': 'incremental_sgd_nystroem_ensemble',
            'n_estimators': args.n_estimators,
            'n_components': args.nystroem_components,
            'timestamp': pd.Timestamp.now().isoformat()
        }
    }

    model_path: Path = Path("outputs") / "model.joblib"
    # compress=0 allows mmap local scoring in downstream tasks
    joblib.dump(artifact, model_path, compress=0)
    mlflow.log_artifact(str(model_path))

    logger.info("Pipeline execution successful. Model artifact persisted.")
    mlflow.end_run()


if __name__ == "__main__":
    main()

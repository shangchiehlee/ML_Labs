"""Cloud training worker for graduate admissions prediction.

Purpose:
    Executes incremental learning via SGDRegressor with adaptive learning rates.
    Uses chunked ingestion and derives regression and classification metrics from
    final epoch predictions.

Architecture:
    - Incremental Learning: partial_fit() on streaming batches.
    - Memory Profile: Per-batch memory is bounded, while final epoch evaluation
      buffers scale with test set size.
    - Evaluation: Regression and classification metrics derived from final epoch
      predictions based on a success threshold.

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
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# --- Shared Utils Import ---
try:
    # When running in Azure, this file is injected into the root of the src dir
    import scalability_utils as utils
except ImportError:
    # Fallback for local dev mode
    sys.path.append(".")
    import scalability_utils as utils  # type: ignore


# --- Logging Infrastructure ---

def setup_forensic_logger() -> logging.Logger:
    """Configures a dedicated forensic logger for the Cloud runtime.

    Returns:
        logging.Logger: Configured logger instance.
    """
    os.makedirs("outputs", exist_ok=True)
    return utils.configure_logging(
        "CloudTrainer", "forensic_audit.log", log_to_file=True
    )


logger: logging.Logger = setup_forensic_logger()


# --- Core Logic ---

def train_epoch(
    data_path: str,
    target_col: str,
    batch_size: int,
    test_ratio: float,
    scaler: StandardScaler,
    regressor: SGDRegressor,
    forensics: utils.StreamForensics,
    is_first: bool,
    accumulate: bool = False,
) -> Tuple[List[float], List[float]]:
    """Trains the model for a single epoch using chunked generators.

    Args:
        data_path: Input URI or file path.
        target_col: Name of the label column.
        batch_size: Row count per partial_fit iteration.
        test_ratio: Fraction of data reserved for incremental validation.
        scaler: Incremental normaliser instance.
        regressor: Incremental SGD regressor instance.
        forensics: Stateful stream auditor.
        is_first: Flag to enable forensic auditing on the first pass.
        accumulate: Flag to collect test predictions for final evaluation.

    Returns:
        Tuple: (True values list, Prediction list) for the test set.
    """
    y_true: List[float] = []
    y_pred: List[float] = []

    # Chunked reader to bound per-batch memory
    # The default iterator allows sequential access without loading whole file
    with pd.read_csv(data_path, chunksize=batch_size) as reader:
        for chunk in reader:
            # Header Normalisation: Strip whitespace and quotes
            chunk.columns = [
                str(c).strip().replace('"', '').replace("'", "")
                for c in chunk.columns
            ]

            if "Serial No." in chunk.columns:
                chunk.drop(columns=["Serial No."], inplace=True)

            if is_first:
                chunk = forensics.audit_chunk(chunk)
                if chunk.empty:
                    continue

            if target_col not in chunk.columns:
                continue

            y_chunk: pd.Series = chunk.pop(target_col)
            x_train, x_test, y_train, y_test = train_test_split(
                chunk, y_chunk, test_size=test_ratio, random_state=42
            )

            # Incremental Update (Partial Fit)
            scaler.partial_fit(x_train)
            x_train_scaled: np.ndarray = scaler.transform(x_train)
            regressor.partial_fit(x_train_scaled, y_train)

            if accumulate:
                x_test_scaled: np.ndarray = scaler.transform(x_test)
                current_preds: np.ndarray = regressor.predict(x_test_scaled)
                y_true.extend(y_test.tolist())
                y_pred.extend(current_preds.tolist())

            del chunk, x_train, x_test, y_train, y_test
            gc.collect()

    return y_true, y_pred


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float
) -> Dict[str, float]:
    """Computes a comprehensive performance scorecard from stream predictions.

    Args:
        y_true: Array of actual continuous values.
        y_pred: Array of predicted continuous values.
        threshold: Decision boundary for classification derivation.

    Returns:
        Dict: Key-value map of regression and classification metrics.
    """
    # Convert continuous regression outputs to binary classification labels
    # based on the business logic threshold (e.g. >0.7 chance = Admit)
    y_true_bin: np.ndarray = (y_true > threshold).astype(int)
    y_pred_bin: np.ndarray = (y_pred > threshold).astype(int)

    metrics: Dict[str, float] = {
        "r2": r2_score(y_true, y_pred),
        "mse": mean_squared_error(y_true, y_pred),
        "mae": mean_absolute_error(y_true, y_pred),
        "accuracy": accuracy_score(y_true_bin, y_pred_bin),
        "f1": f1_score(y_true_bin, y_pred_bin, zero_division=0),
        "precision": precision_score(y_true_bin, y_pred_bin, zero_division=0),
        "recall": recall_score(y_true_bin, y_pred_bin, zero_division=0),
    }

    if len(np.unique(y_true_bin)) > 1:
        metrics["auc"] = roc_auc_score(y_true_bin, y_pred)

    return metrics


def main() -> None:
    """Main training orchestrator entry point."""
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Graduate Admissions Incremental Training Worker"
    )
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--test_train_ratio", type=float, default=0.2)
    parser.add_argument("--learning_rate", type=float, default=0.01)
    parser.add_argument("--max_iter", type=int, default=1000)
    parser.add_argument("--features_json", type=str, required=True)
    parser.add_argument("--registered_model_name", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--threshold", type=float, default=0.7)
    parser.add_argument("--forensic_rows", type=int, default=10_000_000)
    args: argparse.Namespace = parser.parse_args()

    mlflow.start_run()

    try:
        feat_defs: Dict[str, Any] = json.loads(args.features_json)
        target_col: str = feat_defs.get("target", "Chance of Admit")
    except json.JSONDecodeError as e:
        logger.error(f"CRITICAL: Feature schema serialisation error. Error: {e}")
        sys.exit(1)

    # Initialise Incremental Components
    scaler: StandardScaler = StandardScaler()
    regressor: SGDRegressor = SGDRegressor(
        learning_rate="adaptive",
        eta0=args.learning_rate,
        random_state=42,
        max_iter=args.max_iter,
        tol=1e-3
    )
    forensics: utils.StreamForensics = utils.StreamForensics(
        estimated_rows=args.forensic_rows
    )

    logger.info(
        f"Initialising Training: {args.epochs} Epochs | Batch: {args.batch_size}"
    )

    y_test_global: List[float] = []
    y_pred_global: List[float] = []

    try:
        for epoch in range(args.epochs):
            is_last: bool = (epoch == args.epochs - 1)
            # Only accumulate predictions on the final epoch to save memory
            y_true, y_pred = train_epoch(
                data_path=args.data,
                target_col=target_col,
                batch_size=args.batch_size,
                test_ratio=args.test_train_ratio,
                scaler=scaler,
                regressor=regressor,
                forensics=forensics,
                is_first=(epoch == 0),
                accumulate=is_last
            )

            if is_last:
                y_test_global, y_pred_global = y_true, y_pred

            if epoch % 10 == 0:
                logger.info(f"Checkpointed Epoch {epoch}/{args.epochs}")

    except Exception as e:
        logger.error(f"CRITICAL: Learning stream interrupted. Error: {e}")
        sys.exit(1)

    forensics.log_summary()

    if not y_test_global:
        logger.error("CRITICAL: No test data accumulated. Validation failed.")
        sys.exit(1)

    # Deriving Final Performance Metrics
    metrics: Dict[str, float] = calculate_metrics(
        np.array(y_test_global), np.array(y_pred_global), args.threshold
    )

    for k, v in metrics.items():
        mlflow.log_metric(k, v)
        logger.info(f"Metric - {k.upper()}: {v:.4f}")

    # Artifact Persistence
    artifact: Dict[str, Any] = {
        "model": make_pipeline(scaler, regressor),
        "metadata": {
            "sklearn": sklearn.__version__,
            "feature_schema": feat_defs,
            "threshold": args.threshold,
            "timestamp": pd.Timestamp.now().isoformat()
        }
    }

    model_path: Path = Path("outputs") / "model.joblib"
    # compress=0 allows memory-mapped reads during local scoring where supported
    joblib.dump(artifact, model_path, compress=0)
    mlflow.log_artifact(str(model_path))

    logger.info("Pipeline execution successful. Model artifact persisted.")
    mlflow.end_run()


if __name__ == "__main__":
    main()

"""Cloud training worker for incremental student outcome prediction.

Purpose:
    Executes multi-class classification using SGDClassifier with incremental updates
    (partial_fit) over CSV chunks. Uses a cumulative confusion matrix and aggregated
    log loss to compute metrics without storing per-row predictions.

Architecture:
    - Incremental learning with partial_fit on streaming batches.
    - Memory use is bounded by batch size, model state, class count, and forensic state.
    - Evaluation uses a cumulative confusion matrix for precision and recall derivation.

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
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix, log_loss
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

def discover_classes(
    data_path: str,
    target_col: str,
    batch_size: int
) -> np.ndarray:
    """Pre-scans the dataset stream to identify the unique target class definition.

    Ensures that the incremental learner is aware of the global label space
    before the first partial_fit iteration.

    Args:
        data_path: Input URI or path.
        target_col: Label column name.
        batch_size: Iteration batch size for pre-scan.

    Returns:
        np.ndarray: Sorted array of unique class identifiers.
    """
    logger.info("Scanning dataset stream for multi-class definitions...")
    unique_classes: set = set()
    try:
        # Chunked reader to bound memory by batch size during discovery
        with pd.read_csv(data_path, chunksize=batch_size) as reader:
            for chunk in reader:
                # Normalise headers to remove whitespace and apostrophes
                chunk.columns = [
                    str(c).strip().replace('"', '').replace("'", "")
                    for c in chunk.columns
                ]
                if target_col in chunk.columns:
                    unique_classes.update(chunk[target_col].unique())

        all_classes: np.ndarray = np.array(sorted(list(unique_classes)))
        logger.info(f"Global Class Schema Detected: {all_classes}")
        return all_classes
    except Exception as e:
        logger.error(f"CRITICAL: Dataset introspection failed. Error: {e}")
        sys.exit(1)


def calculate_weighted_metrics(
    conf_matrix: np.ndarray,
    support: np.ndarray,
    total_samples: int
) -> Tuple[float, float, float, float]:
    """Derives performance metrics from the cumulative confusion matrix.

    Args:
        conf_matrix: The aggregated confusion matrix (n_classes x n_classes).
        support: True sample count per class.
        total_samples: Global sample count.

    Returns:
        Tuple[float, float, float, float]: (Accuracy, Precision, Recall, F1).
    """
    if total_samples == 0:
        logger.warning("Zero samples processed; metric derivation aborted.")
        return 0.0, 0.0, 0.0, 0.0

    # Global Accuracy
    accuracy: float = float(np.trace(conf_matrix)) / total_samples

    # Per-Class Statistics
    tp: np.ndarray = np.diag(conf_matrix)
    fp: np.ndarray = np.sum(conf_matrix, axis=0) - tp
    fn: np.ndarray = np.sum(conf_matrix, axis=1) - tp

    # Precision/Recall derivation with zero-division safety
    with np.errstate(divide='ignore', invalid='ignore'):
        precision_cls: np.ndarray = np.nan_to_num(tp / (tp + fp))
        recall_cls: np.ndarray = np.nan_to_num(tp / (tp + fn))
        f1_cls: np.ndarray = np.nan_to_num(
            2 * (precision_cls * recall_cls) / (precision_cls + recall_cls)
        )

    # Weighted Aggregation
    total_support: int = int(np.sum(support))
    if total_support == 0:
        return accuracy, 0.0, 0.0, 0.0

    prec_weighted: float = float(np.sum(precision_cls * support)) / total_support
    rec_weighted: float = float(np.sum(recall_cls * support)) / total_support
    f1_weighted: float = float(np.sum(f1_cls * support)) / total_support

    return accuracy, prec_weighted, rec_weighted, f1_weighted


def main() -> None:
    """Main execution entry point for Cloud Training."""
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Student Success Incremental Training Worker"
    )
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--test_train_ratio", type=float, default=0.2)
    parser.add_argument("--learning_rate", type=float, default=0.01)
    parser.add_argument("--max_iter", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=200)
    parser.add_argument("--features_json", type=str, required=True)
    parser.add_argument("--registered_model_name", type=str, default="student_success_model")
    args: argparse.Namespace = parser.parse_args()

    mlflow.start_run()
    logger.info(
        f"Initialising Streaming Context: Batch={args.batch_size} | Epochs={args.max_iter}"
    )

    try:
        feature_defs: Dict[str, Any] = json.loads(args.features_json)
        target_col: str = feature_defs.get("target", "Target")
    except json.JSONDecodeError as e:
        logger.error(f"CRITICAL: Feature schema serialisation error. Error: {e}")
        sys.exit(1)

    # --- Step 1: Label Space Discovery ---
    all_classes: np.ndarray = discover_classes(args.data, target_col, args.batch_size)
    n_classes: int = len(all_classes)

    # --- Step 2: Pipeline Initialisation ---
    scaler: StandardScaler = StandardScaler()
    clf: SGDClassifier = SGDClassifier(
        loss='log_loss',
        learning_rate='adaptive',
        eta0=args.learning_rate,
        penalty='l2',
        random_state=42,
        n_jobs=1
    )

    forensics: utils.StreamForensics = utils.StreamForensics(estimated_rows=5000)

    # Aggregated state for metric calculation with memory proportional to the square of class count
    cumulative_conf_matrix: np.ndarray = np.zeros((n_classes, n_classes), dtype=np.int64)
    cumulative_log_loss: float = 0.0

    # --- Step 3: Streaming Incremental Learning ---
    try:
        for epoch in range(args.max_iter):
            logger.info(f"--- Commencing Epoch {epoch + 1}/{args.max_iter} ---")

            with pd.read_csv(args.data, chunksize=args.batch_size) as reader:
                for chunk in reader:
                    # Header sanitisation
                    chunk.columns = [
                        str(c).strip().replace('"', '').replace("'", "")
                        for c in chunk.columns
                    ]

                    # Deduplication & Null Safety (Baseline established in Epoch 0)
                    if epoch == 0:
                        chunk = forensics.audit_chunk(chunk)

                    if chunk.empty or target_col not in chunk.columns:
                        continue

                    y_chunk: pd.Series = chunk.pop(target_col)

                    # Split for incremental validation
                    x_train, x_test, y_train, y_test = train_test_split(
                        chunk, y_chunk, test_size=args.test_train_ratio, random_state=42
                    )

                    # Normalisation and Online Update
                    scaler.partial_fit(x_train)
                    x_train_scaled: np.ndarray = scaler.transform(x_train)
                    clf.partial_fit(x_train_scaled, y_train, classes=all_classes)

                    # Cumulative Matrix Update (Final Epoch Evaluation)
                    if epoch == args.max_iter - 1:
                        x_test_scaled: np.ndarray = scaler.transform(x_test)
                        y_pred: np.ndarray = clf.predict(x_test_scaled)
                        y_proba: np.ndarray = clf.predict_proba(x_test_scaled)

                        # Accumulate Log Loss (Weighted by batch size)
                        batch_loss = log_loss(y_test, y_proba, labels=all_classes)
                        cumulative_log_loss += batch_loss * len(y_test)
                        
                        # We must ensure labels passed to confusion_matrix match the
                        # global class definition to prevent shape mismatch errors.
                        cumulative_conf_matrix += confusion_matrix(
                            y_test, y_pred, labels=all_classes
                        )

                    # Explicit resource reclamation
                    del chunk, x_train, x_test, y_train, y_test
                    gc.collect()

    except Exception as e:
        logger.error(f"CRITICAL: Learning stream interrupted. Error: {e}")
        sys.exit(1)

    forensics.log_summary()

    # --- Step 4: Forensic Metric Derivation ---
    total_samples = int(np.sum(cumulative_conf_matrix)) # <--- Define total_samples variable
    support: np.ndarray = np.sum(cumulative_conf_matrix, axis=1)
    acc, prec, rec, f1 = calculate_weighted_metrics(
        cumulative_conf_matrix, support, total_samples
    )
    
    # Derive Average Log Loss
    avg_log_loss = cumulative_log_loss / total_samples if total_samples > 0 else 0.0

    logger.info(f"SCORING COMPLETE - Accuracy: {acc:.4f} | F1: {f1:.4f} | Log Loss: {avg_log_loss:.4f}")

    mlflow.log_metrics({
        "accuracy": acc,
        "f1_weighted": f1,
        "precision_weighted": prec,
        "recall_weighted": rec,
        "log_loss": avg_log_loss
    })

    # --- Step 5: Artifact Serialisation ---
    artifact: Dict[str, Any] = {
        'pipeline': make_pipeline(scaler, clf),
        'metadata': {
            'sklearn': sklearn.__version__,
            'classes': all_classes.tolist(),
            'training_mode': 'incremental_sgd_multiclass',
            'timestamp': pd.Timestamp.now().isoformat()
        }
    }

    model_path: Path = Path("outputs") / "model.joblib"
    # compress=0 enables memory-mapped scoring in local inference
    joblib.dump(artifact, model_path, compress=0)
    mlflow.log_artifact(str(model_path))

    logger.info("Pipeline execution successful. Model artifact persisted.")
    mlflow.end_run()


if __name__ == "__main__":
    main()

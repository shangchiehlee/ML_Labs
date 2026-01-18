"""Cloud training worker for incremental NLP learning.

Purpose:
    Trains a text classifier using HashingVectorizer and PassiveAggressiveClassifier
    with chunked CSV ingestion. Writes a model artefact and forensic audit outputs
    derived from the streamed data.

Architecture:
    - Memory Profile: Chunked ingestion bounds per-batch memory, while evaluation
      buffers grow with the number of retained test rows in the final epoch.
    - Time Complexity: Linear per epoch over the input stream.
    - Vectorisation: Stateless hashing to avoid vocabulary storage.

Author: Shane Lee
Licence: MIT
"""

import argparse
import gc
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
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

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

def generate_drivers(
    text_list: List[str],
    labels: List[int],
    top_n: int = 20
) -> pd.DataFrame:
    """Extracts top bi-gram drivers for sentiment segments.

    Args:
        text_list: List of unstructured text samples.
        labels: Binary sentiment labels (0/1).
        top_n: Limit for phrase extraction.

    Returns:
        pd.DataFrame: Sentiment drivers report.
    """
    if not text_list:
        return pd.DataFrame()

    df: pd.DataFrame = pd.DataFrame({'text': text_list, 'label': labels})
    insights: List[Dict[str, Any]] = []

    def extract_top_phrases(corpus: pd.Series, segment: str) -> None:
        if corpus.empty:
            return
        # Use unigrams and bigrams for phrase extraction
        cv: CountVectorizer = CountVectorizer(
            stop_words='english',
            ngram_range=(1, 2),
            max_features=top_n
        )
        try:
            matrix = cv.fit_transform(corpus)
            counts: np.ndarray = matrix.sum(axis=0).A1
            vocab: np.ndarray = cv.get_feature_names_out()

            for word, count in sorted(zip(vocab, counts), key=lambda x: x[1], reverse=True):
                insights.append({'Segment': segment, 'Phrase': word, 'Frequency': count})
        except ValueError:
            # Handle cases with empty vocabulary after stop-word removal
            pass

    extract_top_phrases(df[df['label'] == 1]['text'], "Positive_Drivers")
    extract_top_phrases(df[df['label'] == 0]['text'], "Negative_Drivers")

    return pd.DataFrame(insights)


def process_stream(
    args: argparse.Namespace,
    vectorizer: HashingVectorizer,
    clf: PassiveAggressiveClassifier,
    forensics: utils.StreamForensics,
    audit_path: Path
) -> Tuple[List[int], List[int], List[float], List[str], List[int]]:
    """Executes the incremental learning loop over the input stream.

    Args:
        args: Pipeline configuration.
        vectorizer: Stateless feature extractor.
        clf: Incremental learner.
        forensics: Stream auditing tool.
        audit_path: Path for disk-based audit output.

    Returns:
        Tuple: Accumulated lightweight metrics and reporting buffers.
    """
    classes: np.ndarray = np.array([0, 1])
    y_true: List[int] = []
    y_pred: List[int] = []
    y_conf: List[float] = []

    # Capped insight buffers to bound memory for report extraction
    MAX_INSIGHTS: int = 50000
    insight_text: List[str] = []
    insight_label: List[int] = []

    is_first_write: bool = True

    for epoch in range(args.epochs):
        logger.info(f"--- COMMENCING EPOCH {epoch + 1}/{args.epochs} ---")

        # Chunked reader to bound per-batch memory
        # on_bad_lines='skip' handles potential CSV malformations
        with pd.read_csv(args.data, chunksize=args.batch_size, on_bad_lines='skip') as reader:
            for chunk in reader:
                # Forensic audit on first epoch for null filtering and repeated index checks
                if epoch == 0:
                    chunk = forensics.audit_chunk(chunk)

                if args.text_col not in chunk.columns or args.label_col not in chunk.columns:
                    continue

                chunk = chunk.dropna(subset=[args.text_col, args.label_col])
                if chunk.empty:
                    continue

                # In-line transformation per chunk
                y_chunk: pd.Series = (chunk[args.label_col] > args.threshold).astype(int)
                x_chunk: pd.Series = chunk[args.text_col].astype(str)

                # Train/Test Split on chunk level to simulate streaming holdout
                x_train, x_test, y_train, y_test = train_test_split(
                    x_chunk, y_chunk, test_size=0.2, random_state=42
                )

                # Incremental Update (Partial Fit)
                x_train_vec = vectorizer.transform(x_train)
                clf.partial_fit(x_train_vec, y_train, classes=classes)

                # Final Epoch Evaluation & report extraction
                if epoch == args.epochs - 1:
                    x_test_vec = vectorizer.transform(x_test)
                    preds: np.ndarray = clf.predict(x_test_vec)
                    conf: np.ndarray = clf.decision_function(x_test_vec)

                    # Disk-based audit cache for evaluation outputs
                    pd.DataFrame({
                        'text': x_test,
                        'true': y_test,
                        'pred': preds,
                        'conf': np.abs(conf)
                    }).to_csv(audit_path, mode='a', header=is_first_write, index=False)
                    is_first_write = False

                    y_true.extend(y_test.tolist())
                    y_pred.extend(preds.tolist())
                    y_conf.extend(conf.tolist())

                    if len(insight_text) < MAX_INSIGHTS:
                        limit: int = MAX_INSIGHTS - len(insight_text)
                        insight_text.extend(x_test.iloc[:limit].tolist())
                        insight_label.extend(y_test.iloc[:limit].tolist())

                del chunk
                gc.collect()

    return y_true, y_pred, y_conf, insight_text, insight_label


def generate_artifacts(
    audit_path: Path,
    insight_text: List[str],
    insight_labels: List[int]
) -> None:
    """Produces reporting assets from the audit and driver buffers.

    Args:
        audit_path: Path to the forensic result cache.
        insight_text: Text buffer for driver analysis.
        insight_labels: Label buffer for driver analysis.
    """
    logger.info("Serialising report artefacts...")

    # Priority Alerts: Negative predictions ranked by confidence
    try:
        priority_buffer: List[pd.DataFrame] = []
        if audit_path.exists():
            with pd.read_csv(audit_path, chunksize=50000) as reader:
                for chunk in reader:
                    # Negatives ranked by confidence
                    negatives: pd.DataFrame = chunk[chunk['pred'] == 0]
                    if not negatives.empty:
                        priority_buffer.append(negatives)

            if priority_buffer:
                priority_df: pd.DataFrame = pd.concat(priority_buffer).sort_values(
                    by='conf', ascending=False
                ).head(200)
                priority_path: Path = Path("outputs") / "priority_alerts.csv"
                priority_df.to_csv(priority_path, index=False)
                mlflow.log_artifact(str(priority_path))
    except Exception as e:
        logger.warning(f"Priority alert extraction failed: {e}")

    # Sentiment Drivers: Keyword bi-gram analysis
    drivers_df: pd.DataFrame = generate_drivers(insight_text, insight_labels)
    if not drivers_df.empty:
        drivers_path: Path = Path("outputs") / "sentiment_drivers.csv"
        drivers_df.to_csv(drivers_path, index=False)
        mlflow.log_artifact(str(drivers_path))


def main() -> None:
    """Main training orchestrator."""
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Feedback Classification Incremental Training Worker"
    )
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--n_features", type=int, default=1048576)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=5000)
    parser.add_argument("--text_col", type=str, default="Text")
    parser.add_argument("--label_col", type=str, default="Score")
    parser.add_argument("--threshold", type=int, default=3)
    parser.add_argument("--registered_model_name", type=str, required=True)
    args: argparse.Namespace = parser.parse_args()

    mlflow.start_run()

    # Stateless HashingVectorizer avoids vocabulary storage
    vectorizer: HashingVectorizer = HashingVectorizer(
        n_features=args.n_features,
        alternate_sign=False,
        norm='l2'
    )

    # Online Passive-Aggressive Learner
    clf: PassiveAggressiveClassifier = PassiveAggressiveClassifier(
        C=1.0,
        loss='hinge',
        random_state=42,
        warm_start=True
    )

    forensics: utils.StreamForensics = utils.StreamForensics(estimated_rows=100000)
    audit_path: Path = Path("outputs") / "forensic_audit_full.csv"

    if audit_path.exists():
        audit_path.unlink()

    # Core Execution
    y_true, y_pred, y_conf, insight_text, insight_label = process_stream(
        args, vectorizer, clf, forensics, audit_path
    )

    forensics.log_summary()

    if y_true:
        # Metric Derivation
        acc: float = accuracy_score(y_true, y_pred)
        f1: float = f1_score(y_true, y_pred, average='weighted')
        prec: float = precision_score(y_true, y_pred, average='weighted')
        rec: float = recall_score(y_true, y_pred, average='weighted')
        try:
            auc: float = roc_auc_score(y_true, y_conf)
        except ValueError:
            auc = 0.0

        logger.info(f"FINAL PERFORMANCE: ACC={acc:.4f}, F1={f1:.4f}, AUC={auc:.4f}, PREC={prec:.4f}, REC={rec:.4f}")
        mlflow.log_metrics({
            "accuracy": acc,
            "f1_weighted": f1,
            "auc_roc": auc,
            "precision_weighted": prec,
            "recall_weighted": rec
        })

        # Reporting
        generate_artifacts(audit_path, insight_text, insight_label)

        # Pipeline Persistence
        pipeline = make_pipeline(vectorizer, clf)
        artifact: Dict[str, Any] = {
            'pipeline': pipeline,
            'metadata': {
                'sklearn': sklearn.__version__,
                'metrics': {'accuracy': acc, 'f1': f1},
                'timestamp': pd.Timestamp.now().isoformat()
            }
        }
        
        # compress=0 allows memory-mapped reads during local scoring where supported
        model_path: Path = Path("outputs") / "model.joblib"
        joblib.dump(artifact, model_path, compress=0)
        mlflow.log_artifact(str(model_path))
        mlflow.log_artifact(str(audit_path))

    logger.info("Pipeline execution successful. Model artifact persisted.")
    mlflow.end_run()


if __name__ == "__main__":
    main()

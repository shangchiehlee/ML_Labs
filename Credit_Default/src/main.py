"""
Script Name: main.py
Author: Shane Lee
Description: Cloud Training Worker. Executes the training pipeline on Azure Compute.
             Implements Nystroem Kernel Approximation and a Streaming Ensemble (Bagging) of SGD Classifiers to resolve non-linear decision boundaries within memory constraints.
"""

import os
import sys
import logging
import argparse
import joblib
import json
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import gc
import sklearn  # <--- FIXED: Added root import for version checking

from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_approximation import Nystroem
from sklearn.metrics import (
    accuracy_score, 
    f1_score, 
    roc_auc_score, 
    precision_score, 
    recall_score
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

# --- Shared Utils Import ---
try:
    import scalability_utils as utils
except ImportError:
    print("CRITICAL: scalability_utils.py not found in source directory.")
    sys.exit(1)

# --- Forensic Logging Configuration ---

os.makedirs("outputs", exist_ok=True)

log_formatter = logging.Formatter(
    '%(asctime)s - %(levelname)s - %(message)s', 
    datefmt='%Y-%m-%d %H:%M:%S'
)

file_handler = logging.FileHandler("outputs/forensic_audit.log")
file_handler.setFormatter(log_formatter)
file_handler.setLevel(logging.INFO)

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(log_formatter)
console_handler.setLevel(logging.INFO)

logger = logging.getLogger("ForensicLogger")
logger.setLevel(logging.INFO)
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# --- Main Execution ---

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, help="path to input data")
    parser.add_argument("--test_train_ratio", type=float, default=0.2)
    parser.add_argument("--learning_rate", type=float, default=0.01)
    parser.add_argument("--max_iter", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=50000, help="Batch size for streaming")
    parser.add_argument("--features_json", type=str, help="JSON string of feature definitions")
    parser.add_argument("--registered_model_name", type=str, help="model name")
    args = parser.parse_args()
   
    mlflow.start_run()
    logger.info(f"Loading data stream from: {args.data}")
    logger.info(f"Training Configuration - Epochs: {args.max_iter}, LR: {args.learning_rate}, Batch: {args.batch_size}")
    
    try:
        feature_defs = json.loads(args.features_json)
        target_col = feature_defs.get("target", "default payment next month")
        logger.info(f"Feature Schema Loaded.")
    except Exception as e:
        logger.error(f"CRITICAL: Failed to load schema. Error: {e}")
        sys.exit(1)

    BATCH_SIZE = args.batch_size

    # --- Pre-scan: Class Weights & Discovery ---
    
    logger.info("Scanning full dataset to determine Classes and Weights...")
    
    class_counter = {}
    total_samples = 0
    
    scan_reader = pd.read_csv(args.data, header=1, index_col=0, chunksize=BATCH_SIZE, low_memory=False)
    
    try:
        for chunk in scan_reader:
            if target_col in chunk.columns:
                y_chunk = chunk[target_col].dropna()
                counts = y_chunk.value_counts().to_dict()
                for cls, count in counts.items():
                    class_counter[cls] = class_counter.get(cls, 0) + count
                    total_samples += count
            del chunk
    except Exception as e:
        logger.error(f"CRITICAL: Pre-scan failed. Error: {e}")
        sys.exit(1)
    
    if not class_counter:
        logger.error("CRITICAL: No target data found during scan.")
        sys.exit(1)

    # Compute Weights: n_samples / (n_classes * count_i)
    n_classes = len(class_counter)
    all_classes = np.array(sorted(class_counter.keys()))
    class_weights = {}
    
    for cls in all_classes:
        count = class_counter[cls]
        weight = total_samples / (n_classes * count)
        class_weights[cls] = weight
        
    logger.info(f"Classes Detected: {all_classes}")
    logger.info(f"Calculated Class Weights: {class_weights}")

    # --- Pipeline Construction ---
    
    # ARCHITECTURAL UPGRADE: Ensemble + Nystroem
    # 1. StandardScaler: Essential for Kernel methods.
    # 2. Nystroem: Uses a subset of data as landmarks to approximate RBF kernel. 
    #    More accurate than RBFSampler for structured data.
    
    scaler = StandardScaler()
    # n_components=1200 provides a rich feature space.
    feature_map = Nystroem(gamma=0.1, n_components=1200, random_state=42)
    
    # ENSEMBLE (BAGGING):
    # We initialize 7 independent models with different random seeds.
    # This reduces variance and improves generalization (Bagging effect).
    N_ESTIMATORS = 7
    models = []
    
    for i in range(N_ESTIMATORS):
        clf = SGDClassifier(
            loss='log_loss',
            learning_rate='adaptive',
            eta0=0.01, 
            alpha=0.0001,
            penalty='l2', # L2 works well with Nystroem
            class_weight=class_weights,
            random_state=42 + i, # Crucial: Different seed for each model
            n_jobs=1 # We parallelize via Azure, keep local single-threaded to avoid contention
        )
        models.append(clf)
    
    logger.info(f"Initialized Ensemble with {N_ESTIMATORS} estimators.")
    
    forensics = utils.StreamForensics(estimated_rows=10_000_000)
    
    y_test_global = []
    y_pred_global = [] # Will hold ensemble votes
    y_prob_global = [] # Will hold ensemble average probabilities
    
    # --- Streaming Training Loop (Multi-Epoch) ---
    
    try:
        for epoch in range(args.max_iter):
            logger.info(f"--- Starting Epoch {epoch + 1}/{args.max_iter} ---")
            
            reader = pd.read_csv(args.data, header=1, index_col=0, chunksize=BATCH_SIZE, low_memory=False)
            
            for chunk in reader:
                
                # 1. Forensic Audit (First epoch only)
                if epoch == 0:
                    chunk = forensics.audit_chunk(chunk)
                else:
                    chunk = chunk.dropna()
                
                if chunk.empty:
                    continue

                # 2. Split Features/Target
                if target_col not in chunk.columns:
                    continue
                    
                y_chunk = chunk.pop(target_col)
                X_chunk = chunk
                
                # 3. Train/Test Split
                X_train, X_test, y_train, y_test = train_test_split(
                    X_chunk, y_chunk, test_size=args.test_train_ratio, random_state=42
                )
                
                # 4. Incremental Transformation
                scaler.partial_fit(X_train)
                X_train_scaled = scaler.transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                # Fit Nystroem ONCE on the first chunk of the first epoch
                if epoch == 0 and not hasattr(feature_map, 'components_'):
                    # Nystroem needs 'fit', not 'partial_fit'. 
                    # We fit on the first batch (subset) which acts as the landmark set.
                    feature_map.fit(X_train_scaled)
                
                # Safety: If for some reason it wasn't fitted (e.g. empty first chunk), try again
                if not hasattr(feature_map, 'components_'):
                     feature_map.fit(X_train_scaled)

                X_train_mapped = feature_map.transform(X_train_scaled)
                X_test_mapped = feature_map.transform(X_test_scaled)
                
                # 5. Incremental Training (Ensemble)
                for model in models:
                    model.partial_fit(X_train_mapped, y_train, classes=all_classes)
                
                # 6. Accumulate Test Results (Final Epoch Only)
                if epoch == args.max_iter - 1:
                    y_test_global.extend(y_test)
                    
                    # Ensemble Prediction: Average the probabilities
                    batch_probs = []
                    for model in models:
                        batch_probs.append(model.predict_proba(X_test_mapped)[:, 1])
                    
                    # Average probability across all models
                    avg_probs = np.mean(batch_probs, axis=0)
                    
                    # Convert to class predictions (Threshold 0.5)
                    avg_preds = (avg_probs >= 0.5).astype(int)
                    
                    y_pred_global.extend(avg_preds)
                    y_prob_global.extend(avg_probs)
                
                del chunk, X_train, X_test, y_train, y_test, X_train_mapped, X_test_mapped
                gc.collect()

    except Exception as e:
        logger.error(f"CRITICAL: Stream processing failed. Error: {e}")
        sys.exit(1)

    forensics.log_summary()

    # --- Evaluation ---

    logger.info("Calculating Final Ensemble Metrics...")
    
    acc = accuracy_score(y_test_global, y_pred_global)
    f1 = f1_score(y_test_global, y_pred_global, average='weighted')
    prec = precision_score(y_test_global, y_pred_global, average='weighted', zero_division=0)
    rec = recall_score(y_test_global, y_pred_global, average='weighted', zero_division=0)
    
    try:
        auc = roc_auc_score(y_test_global, y_prob_global)
    except ValueError:
        auc = 0.0
        logger.warning("AUC could not be calculated.")

    logger.info(f"Metrics - Acc: {acc:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}, Recall: {rec:.4f}")
    
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("precision", prec)
    mlflow.log_metric("recall", rec)
    mlflow.log_metric("auc_roc", auc)
    
    # --- Artifact Serialisation ---

    logger.info("Serialising model pipeline...")
    
    # We save the preprocessor and the list of models separately.
    # This allows flexible inference (loading models and averaging).
    
    artifact = {
        'preprocessor': make_pipeline(scaler, feature_map),
        'ensemble_models': models,
        'metadata': {
            'python': sys.version,
            'sklearn': sklearn.__version__,
            'feature_schema': feature_defs,
            'training_mode': 'incremental_sgd_nystroem_ensemble',
            'n_estimators': N_ESTIMATORS,
            'epochs': args.max_iter,
            'class_weights': class_weights,
            'created_at': pd.Timestamp.now().isoformat()
        }
    }

    model_path = "outputs/model.joblib"
    # CRITICAL: compress=0 is required for mmap_mode='r' to work in inference.
    joblib.dump(artifact, model_path, compress=0)

    mlflow.log_artifact("outputs/forensic_audit.log")
    mlflow.log_artifact(model_path)

    logger.info("Job Complete.")
    mlflow.end_run()

if __name__ == "__main__":
    main()
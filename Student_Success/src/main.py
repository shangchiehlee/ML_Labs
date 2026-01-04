"""
Script Name: main.py
Author: Shane Lee
Description: Cloud Training Worker. Executes Multi-Class Classification using
             SGDClassifier with Incremental Learning (partial_fit).
             Implements Micro-Batching and Header Cleaning for robust stream processing.
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
import sklearn

from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, 
    f1_score, 
    precision_score, 
    recall_score
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

try:
    import scalability_utils as utils
except ImportError:
    print("CRITICAL: scalability_utils.py not found.")
    sys.exit(1)

# --- Forensic Logging ---
os.makedirs("outputs", exist_ok=True)
log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
file_handler = logging.FileHandler("outputs/forensic_audit.log")
file_handler.setFormatter(log_formatter)
file_handler.setLevel(logging.INFO)
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(log_formatter)
logger = logging.getLogger("ForensicLogger")
logger.setLevel(logging.INFO)
logger.addHandler(file_handler)
logger.addHandler(console_handler)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, help="path to input data")
    parser.add_argument("--test_train_ratio", type=float, default=0.2)
    parser.add_argument("--learning_rate", type=float, default=0.01)
    parser.add_argument("--max_iter", type=int, default=50)
    # CRITICAL: Accept batch_size argument from job submission
    parser.add_argument("--batch_size", type=int, default=500)
    parser.add_argument("--features_json", type=str, help="JSON string of feature definitions")
    parser.add_argument("--registered_model_name", type=str, help="model name")
    args = parser.parse_args()
   
    mlflow.start_run()
    logger.info(f"Loading data stream from: {args.data} (Batch Size: {args.batch_size})")
    
    try:
        feature_defs = json.loads(args.features_json)
        target_col = feature_defs.get("target", "Target")
    except Exception as e:
        logger.error(f"CRITICAL: Failed to load schema. Error: {e}")
        sys.exit(1)

    BATCH_SIZE = args.batch_size
    
    # --- Pre-scan: Class Discovery ---
    logger.info("Scanning dataset for Multi-Class definitions...")
    unique_classes = set()
    try:
        scan_reader = pd.read_csv(args.data, chunksize=BATCH_SIZE)
        for chunk in scan_reader:
            # Clean headers: Remove tabs, spaces, quotes, AND apostrophes
            chunk.columns = [c.strip().replace('"', '').replace("'", "") for c in chunk.columns]
            if target_col in chunk.columns:
                unique_classes.update(chunk[target_col].unique())
        
        all_classes = np.array(sorted(list(unique_classes)))
        logger.info(f"Classes Detected ({len(all_classes)}): {all_classes}")
    except Exception as e:
        logger.error(f"CRITICAL: Class discovery failed. Error: {e}")
        sys.exit(1)

    # --- Pipeline Construction ---
    scaler = StandardScaler()
    clf = SGDClassifier(
        loss='log_loss', 
        learning_rate='adaptive',
        eta0=args.learning_rate,
        penalty='l2',
        random_state=42,
        n_jobs=1
    )
    
    forensics = utils.StreamForensics(estimated_rows=5000)
    y_test_global = []
    y_pred_global = []
    
    # --- Streaming Training Loop ---
    try:
        for epoch in range(args.max_iter):
            logger.info(f"--- Starting Epoch {epoch + 1}/{args.max_iter} ---")
            reader = pd.read_csv(args.data, chunksize=BATCH_SIZE)
            
            for chunk in reader:
                # Clean headers: Remove tabs, spaces, quotes, AND apostrophes
                chunk.columns = [c.strip().replace('"', '').replace("'", "") for c in chunk.columns]

                if epoch == 0:
                    chunk = forensics.audit_chunk(chunk)
                
                if chunk.empty: continue

                if target_col not in chunk.columns: continue
                    
                y_chunk = chunk.pop(target_col)
                X_chunk = chunk
                
                X_train, X_test, y_train, y_test = train_test_split(
                    X_chunk, y_chunk, test_size=args.test_train_ratio, random_state=42
                )
                
                scaler.partial_fit(X_train)
                X_train_scaled = scaler.transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                clf.partial_fit(X_train_scaled, y_train, classes=all_classes)
                
                if epoch == args.max_iter - 1:
                    y_test_global.extend(y_test)
                    y_pred_global.extend(clf.predict(X_test_scaled))
                
                del chunk, X_train, X_test, y_train, y_test
                gc.collect()

    except Exception as e:
        logger.error(f"CRITICAL: Stream processing failed. Error: {e}")
        sys.exit(1)

    forensics.log_summary()

    # --- Evaluation ---
    logger.info("Calculating Final Multi-Class Metrics...")
    
    acc = accuracy_score(y_test_global, y_pred_global)
    f1 = f1_score(y_test_global, y_pred_global, average='weighted')
    prec = precision_score(y_test_global, y_pred_global, average='weighted', zero_division=0)
    rec = recall_score(y_test_global, y_pred_global, average='weighted', zero_division=0)
    
    logger.info(f"Metrics - Accuracy: {acc:.4f}, F1 (Weighted): {f1:.4f}")
    
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("f1_weighted", f1)
    mlflow.log_metric("precision_weighted", prec)
    mlflow.log_metric("recall_weighted", rec)
    
    # --- Artifact Serialisation ---
    logger.info("Serialising model pipeline...")
    final_pipeline = make_pipeline(scaler, clf)
    
    artifact = {
        'pipeline': final_pipeline,
        'metadata': {
            'python': sys.version,
            'sklearn': sklearn.__version__,
            'classes': all_classes.tolist(),
            'training_mode': 'incremental_sgd_multiclass',
            'created_at': pd.Timestamp.now().isoformat()
        }
    }

    model_path = "outputs/model.joblib"
    # CRITICAL: compress=0 enables mmap_mode='r' for efficient inference loading
    joblib.dump(artifact, model_path, compress=0)

    mlflow.log_artifact("outputs/forensic_audit.log")
    mlflow.log_artifact(model_path)
    logger.info("Job Complete.")
    mlflow.end_run()

if __name__ == "__main__":
    main()
"""
Script Name: main.py
Author: Shane Lee
Description: Cloud Training Worker. Executes the SGDRegressor algorithm using
             incremental learning (partial_fit) and epoch-based iteration to ensure
             convergence. Calculates and logs dual-task performance metrics
             (Regression and Classification) to MLflow for observability.
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

from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    r2_score, 
    mean_squared_error, 
    mean_absolute_error,
    accuracy_score,
    roc_auc_score,
    f1_score,
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
log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

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
    parser.add_argument("--features_json", type=str, help="JSON string of feature definitions")
    parser.add_argument("--registered_model_name", type=str, help="model name")
    args = parser.parse_args()
   
    mlflow.start_run()
    logger.info(f"Loading data stream from: {args.data}")
    
    try:
        feature_defs = json.loads(args.features_json)
        target_col = feature_defs.get("target", "Chance of Admit")
    except Exception as e:
        logger.error(f"CRITICAL: Failed to load schema. Error: {e}")
        sys.exit(1)

    # --- Pipeline Construction (Incremental) ---
    
    scaler = StandardScaler()
    
    # OPTIMISATION: Use 'adaptive' learning rate for better convergence on small data
    regressor = SGDRegressor(
        learning_rate='adaptive',
        eta0=args.learning_rate,
        random_state=42,
        max_iter=1000,
        tol=1e-3
    )
    
    forensics = utils.StreamForensics(estimated_rows=100_000)
    
    # Global accumulators for final evaluation
    # Note: In a true streaming scenario (TB of data), we would evaluate per chunk.
    # For this dataset, we accumulate to calculate global metrics.
    y_test_global = []
    y_pred_global = []
    
    # SCALABILITY CONFIGURATION
    # For small datasets (<100k rows), SGD requires multiple passes (Epochs) to converge.
    # For massive datasets (>10M rows), a single pass (Epoch=1) is usually sufficient.
    NUM_EPOCHS = 100 
    BATCH_SIZE = 50000
    
    logger.info(f"Starting Training Loop: {NUM_EPOCHS} Epochs.")

    try:
        for epoch in range(NUM_EPOCHS):
            # Reset reader for each epoch
            reader = pd.read_csv(args.data, chunksize=BATCH_SIZE)
            chunk_idx = 0
            
            for chunk in reader:
                chunk_idx += 1
                chunk.columns = chunk.columns.str.strip()
                
                if 'Serial No.' in chunk.columns:
                    chunk.drop(columns=['Serial No.'], inplace=True)
                
                # 1. Forensic Audit (Only run on first epoch to avoid log spam)
                if epoch == 0:
                    chunk = forensics.audit_chunk(chunk)
                    if chunk.empty:
                        continue

                # 2. Split Features/Target
                y_chunk = chunk.pop(target_col)
                X_chunk = chunk
                
                # 3. Train/Test Split
                # Note: random_state ensures the split is consistent across epochs
                X_train, X_test, y_train, y_test = train_test_split(
                    X_chunk, y_chunk, test_size=args.test_train_ratio, random_state=42
                )
                
                # 4. Incremental Scaling
                scaler.partial_fit(X_train)
                X_train_scaled = scaler.transform(X_train)
                
                # 5. Incremental Training
                regressor.partial_fit(X_train_scaled, y_train)
                
                # 6. Evaluation Accumulation (Only on final epoch)
                if epoch == NUM_EPOCHS - 1:
                    X_test_scaled = scaler.transform(X_test)
                    y_test_global.extend(y_test)
                    y_pred_global.extend(regressor.predict(X_test_scaled))
                
                del chunk, X_train, X_test, y_train, y_test
                gc.collect()
            
            if epoch % 10 == 0:
                logger.info(f"Completed Epoch {epoch}/{NUM_EPOCHS}")

    except Exception as e:
        logger.error(f"CRITICAL: Stream processing failed. Error: {e}")
        sys.exit(1)

    forensics.log_summary()

    # --- Evaluation ---

    logger.info("Calculating Final Metrics...")
    
    y_true = np.array(y_test_global)
    y_pred = np.array(y_pred_global)
    
    # 1. Regression Metrics (Primary)
    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    
    # 2. Classification Metrics (Secondary - Derived via Threshold)
    # Business Logic: If Chance of Admit > 0.7, classify as 'Admit' (1)
    # Note: Adjusted threshold to 0.7 to reflect realistic admission standards
    THRESHOLD = 0.7
    y_true_bin = (y_true > THRESHOLD).astype(int)
    y_pred_bin = (y_pred > THRESHOLD).astype(int)
    
    acc = accuracy_score(y_true_bin, y_pred_bin)
    f1 = f1_score(y_true_bin, y_pred_bin, zero_division=0)
    prec = precision_score(y_true_bin, y_pred_bin, zero_division=0)
    rec = recall_score(y_true_bin, y_pred_bin, zero_division=0)
    
    try:
        auc = roc_auc_score(y_true_bin, y_pred)
    except ValueError:
        auc = 0.0
        logger.warning("AUC could not be calculated (likely single class in test set).")

    logger.info(f"--- Regression Metrics ---")
    logger.info(f"R2: {r2:.4f} | MSE: {mse:.4f} | RMSE: {rmse:.4f} | MAE: {mae:.4f}")
    
    logger.info(f"--- Classification Metrics (Threshold: {THRESHOLD}) ---")
    logger.info(f"Accuracy: {acc:.4f} | AUC: {auc:.4f} | F1: {f1:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f}")
    
    # Log to MLflow
    mlflow.log_metric("r2_score", r2)
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("auc_roc", auc)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("precision", prec)
    mlflow.log_metric("recall", rec)
    
    # --- Artifact Serialisation ---

    logger.info("Serialising model pipeline...")
    final_pipeline = make_pipeline(scaler, regressor)
    
    import sklearn
    artifact = {
        'model': final_pipeline,
        'metadata': {
            'python': sys.version,
            'sklearn': sklearn.__version__,
            'feature_schema': feature_defs,
            'training_mode': 'incremental_sgd_epochs',
            'created_at': pd.Timestamp.now().isoformat(),
            'metrics_threshold': THRESHOLD
        }
    }

    model_path = "outputs/model.joblib"
    joblib.dump(artifact, model_path, compress=0)

    mlflow.log_artifact("outputs/forensic_audit.log")
    mlflow.log_artifact(model_path)

    logger.info("Job Complete.")
    mlflow.end_run()

if __name__ == "__main__":
    main()
"""
Script Name: 04_local_model_scoring.py
Author: Shane Lee
Description: Inference Engine. Deserialises the model artifact using memory mapping.
             Executes risk scoring using parallel batch processing.
             Aggregates predictions from the ensemble via soft voting and enforces output schema validation.
"""

import os
import sys
import json
import logging
import pandas as pd
import numpy as np
import joblib
import gc
import sklearn
from concurrent.futures import ProcessPoolExecutor, as_completed, wait, FIRST_COMPLETED

# --- Local Utils Import ---
import scalability_utils as utils

# --- Configuration & Logging ---

try:
    with open("config.json", "r") as f:
        config = json.load(f)
    
    # Validate Schema Version
    utils.validate_config_version(config, expected_version="1.0")

    ops_settings = config.get("operational_settings", {})
    N_JOBS = ops_settings.get("inference_n_jobs", 2)
    DRY_RUN_ROWS = ops_settings.get("dry_run_rows", None)
    SCHEMA_CONFIG = config.get("schema_expectations", {})
    OUTPUT_SCHEMA = config.get("output_schema_expectations", {})
    # Prioritise config batch size
    CONFIG_BATCH_SIZE = ops_settings.get("batch_size", None)
except Exception as e:
    print(f"Config load failed: {e}")
    sys.exit(1)

logger = utils.configure_logging("ScoringForensics", "scoring_ops.log", config)

# Determine Batch Size: Config > Dynamic
if CONFIG_BATCH_SIZE:
    BATCH_SIZE = CONFIG_BATCH_SIZE
    logger.info(f"Using Configured Batch Size: {BATCH_SIZE}")
else:
    BATCH_SIZE = utils.calculate_optimal_batch_size(n_jobs=N_JOBS, safety_factor=0.6, row_size_estimate=1500)
    logger.info(f"Using Dynamic Batch Size: {BATCH_SIZE}")

INPUT_FILE = "inputs/scoring_input_data.csv"
OUTPUT_FILE = "outputs/scoring_results_prediction.csv"

# --- Model Location ---

try:
    with open("logs/latest_job.txt", "r") as f:
        job_name = f.read().strip()
    logger.info(f"Targeting Job ID: {job_name}")
except FileNotFoundError:
    logger.error("CRITICAL: State file 'logs/latest_job.txt' missing.")
    sys.exit(1)

search_dir = f"./downloaded_artifacts/{job_name}"
model_path = None

for root, dirs, files in os.walk(search_dir):
    if "model.joblib" in files:
        model_path = os.path.join(root, "model.joblib")
        break

if not model_path:
    logger.error(f"CRITICAL: 'model.joblib' not found in {search_dir}")
    sys.exit(1)

# --- Global Model Variables (Worker Scope) ---
global_preprocessor = None
global_ensemble_models = None
global_single_model = None
is_ensemble_mode = False

def init_worker(path):
    """
    Initialises the worker process.
    Detects whether the artifact is a Single Model or an Ensemble.
    """
    global global_preprocessor
    global global_ensemble_models
    global global_single_model
    global is_ensemble_mode
    
    try:
        # Attempt mmap load
        artifact = joblib.load(path, mmap_mode='r')
    except Exception:
        # Fallback for compressed files
        artifact = joblib.load(path)
        
    # Detect Architecture
    if 'ensemble_models' in artifact:
        is_ensemble_mode = True
        global_preprocessor = artifact['preprocessor']
        global_ensemble_models = artifact['ensemble_models']
    elif 'model' in artifact:
        is_ensemble_mode = False
        global_single_model = artifact['model']
    else:
        # Fallback: Assume the artifact itself is the model (legacy)
        is_ensemble_mode = False
        global_single_model = artifact

def validate_output_schema(df, schema_config):
    """
    Validates the output dataframe against the contract in config.json.
    """
    if not schema_config:
        return df

    required = schema_config.get("required_columns", [])
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Output Schema Violation: Missing columns {missing}")
    
    return df

def validate_and_process_batch(chunk, input_schema, output_schema):
    """
    Worker function: Validates input, transforms features, predicts (Ensemble/Single), 
    and validates output.
    """
    try:
        # 1. Input Validation
        if input_schema:
            required = input_schema.get("required_columns", [])
            # Check for missing features (ignoring target if present in schema but not input)
            missing = [c for c in required if c not in chunk.columns and c != "default payment next month"]
            if missing:
                return Exception(f"Input Schema Violation: Missing {missing}")

        # 2. Prediction Logic
        if is_ensemble_mode:
            # A. Transform Features (Scaler + Nystroem)
            # Note: preprocessor is a Pipeline, so it handles the sequence
            X_trans = global_preprocessor.transform(chunk)
            
            # B. Ensemble Voting (Soft Vote)
            # Collect probabilities from all estimators
            probas_stack = []
            for model in global_ensemble_models:
                probas_stack.append(model.predict_proba(X_trans)[:, 1])
            
            # Average the probabilities across models
            avg_probs = np.mean(probas_stack, axis=0)
            chunk["Probability of Default"] = avg_probs
            
        else:
            # Legacy Single Model
            chunk["Probability of Default"] = global_single_model.predict_proba(chunk)[:, 1]
        
        # 3. Output Validation
        validate_output_schema(chunk, output_schema)
        
        return chunk
    except Exception as e:
        return e 

# --- Main Execution ---

def main():
    logger.info("Deserialising model artifact for validation...")
    try:
        # Initial load for metadata check (without mmap)
        artifact = joblib.load(model_path)
        metadata = artifact.get('metadata', {})
        
        # Version Handshake
        train_python = metadata.get("python", "Unknown")
        train_sklearn = metadata.get("sklearn", "Unknown")
        local_python = sys.version
        local_sklearn = sklearn.__version__
        
        logger.info(f"Model Metadata - Sklearn: {train_sklearn}, Python: {train_python}")
        logger.info(f"Training Mode: {metadata.get('training_mode', 'Unknown')}")
        
        if train_sklearn != local_sklearn:
            logger.warning(f"VERSION MISMATCH: Model trained on Sklearn {train_sklearn}, running on {local_sklearn}. Behaviour may be undefined.")
        
        del artifact
        gc.collect()
    except Exception as e:
        logger.error(f"CRITICAL: Model corruption. Error: {e}")
        sys.exit(1)

    logger.info(f"Initialising Parallel Processing on: {INPUT_FILE}")
    if DRY_RUN_ROWS:
        logger.info(f"DRY RUN ENABLED: Processing limited to {DRY_RUN_ROWS} rows.")
    
    if not os.path.exists(INPUT_FILE):
        logger.error(f"CRITICAL: Input file missing.")
        sys.exit(1)

    os.makedirs("outputs", exist_ok=True)
    if os.path.exists(OUTPUT_FILE):
        os.remove(OUTPUT_FILE)

    chunk_iterator = pd.read_csv(INPUT_FILE, index_col=0, chunksize=BATCH_SIZE)
    
    total_processed = 0
    MAX_QUEUE_SIZE = N_JOBS * 2
    
    # Initialize Workers with the model path
    with ProcessPoolExecutor(max_workers=N_JOBS, initializer=init_worker, initargs=(model_path,)) as executor:
        futures = set()
        
        for chunk in chunk_iterator:
            # Dry Run Check
            if DRY_RUN_ROWS and total_processed >= DRY_RUN_ROWS:
                logger.info(f"DRY RUN: Limit of {DRY_RUN_ROWS} rows reached. Stopping submission.")
                break

            if len(futures) >= MAX_QUEUE_SIZE:
                done, futures = wait(futures, return_when=FIRST_COMPLETED)
                
                for future in done:
                    result = future.result()
                    if isinstance(result, Exception):
                        logger.error(f"Worker failed: {result}")
                    else:
                        write_header = (total_processed == 0)
                        # FIX: float_format prevents scientific notation
                        result.to_csv(OUTPUT_FILE, mode='a', header=write_header, float_format='%.8f')
                        total_processed += len(result)
                        logger.info(f"Batch processed. Total rows: {total_processed}")
                        del result
                        gc.collect()

            future = executor.submit(validate_and_process_batch, chunk, SCHEMA_CONFIG, OUTPUT_SCHEMA)
            futures.add(future)

        # Process remaining futures
        for future in as_completed(futures):
            result = future.result()
            if isinstance(result, Exception):
                logger.error(f"Worker failed: {result}")
            else:
                write_header = (total_processed == 0)
                result.to_csv(OUTPUT_FILE, mode='a', header=write_header, float_format='%.8f')
                total_processed += len(result)
                logger.info(f"Batch processed. Total rows: {total_processed}")
                del result
                gc.collect()

    if total_processed > 0:
        logger.info(f"--- SCORING COMPLETE ---")
        logger.info(f"Results saved to: {OUTPUT_FILE}")
    else:
        logger.error("CRITICAL: No rows processed. Check logs for worker errors.")

if __name__ == "__main__":
    main()
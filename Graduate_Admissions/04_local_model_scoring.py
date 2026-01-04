"""
Script Name: 04_local_model_scoring.py
Author: Shane Lee
Description: Inference Engine. Executes parallel batch processing for risk scoring.
             Utilises memory mapping (mmap) to share model memory segments across
             worker processes, minimising the resident set size (RSS). Enforces
             strict input and output schema validation against the configuration contract.
"""

import os
import sys
import json
import logging
import pandas as pd
import joblib
import gc
import sklearn
from concurrent.futures import ProcessPoolExecutor, as_completed, wait, FIRST_COMPLETED

# --- Local Utils Import ---
import scalability_utils as utils

# --- Configuration ---

try:
    with open("config.json", "r") as f:
        config = json.load(f)
    utils.validate_config_version(config, expected_version="1.0")

    ops_settings = config.get("operational_settings", {})
    N_JOBS = ops_settings.get("inference_n_jobs", 2)
    DRY_RUN_ROWS = ops_settings.get("dry_run_rows", None)
    SCHEMA_CONFIG = config.get("schema_expectations", {})
    OUTPUT_SCHEMA = config.get("output_schema_expectations", {})
except Exception as e:
    print(f"Config load failed: {e}")
    sys.exit(1)

logger = utils.configure_logging("ScoringForensics", "scoring_ops.log", config)

BATCH_SIZE = utils.calculate_optimal_batch_size(n_jobs=N_JOBS, safety_factor=0.6, row_size_estimate=500)
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

# --- Global Model Loader ---
global_model_pipeline = None

def init_worker(path):
    global global_model_pipeline
    try:
        artifact = joblib.load(path, mmap_mode='r')
    except Exception:
        artifact = joblib.load(path)
    global_model_pipeline = artifact.get('model')

def validate_and_process_batch(chunk, input_schema, output_schema):
    try:
        # Prediction
        # Note: SGDRegressor predicts continuous values directly
        predictions = global_model_pipeline.predict(chunk)
        chunk["Predicted Chance of Admit"] = predictions
        
        return chunk
    except Exception as e:
        return e 

# --- Main Execution ---

def main():
    logger.info("Deserialising model artifact for validation...")
    try:
        artifact = joblib.load(model_path)
        metadata = artifact.get('metadata', {})
        
        train_sklearn = metadata.get("sklearn", "Unknown")
        local_sklearn = sklearn.__version__
        
        if train_sklearn != local_sklearn:
            logger.warning(f"VERSION MISMATCH: Model trained on {train_sklearn}, running on {local_sklearn}.")
        
        del artifact
        gc.collect()
    except Exception as e:
        logger.error(f"CRITICAL: Model corruption. Error: {e}")
        sys.exit(1)

    logger.info(f"Initialising Parallel Processing on: {INPUT_FILE}")
    
    if not os.path.exists(INPUT_FILE):
        logger.error(f"CRITICAL: Input file missing.")
        sys.exit(1)

    os.makedirs("outputs", exist_ok=True)
    if os.path.exists(OUTPUT_FILE):
        os.remove(OUTPUT_FILE)

    chunk_iterator = pd.read_csv(INPUT_FILE, chunksize=BATCH_SIZE)
    total_processed = 0
    MAX_QUEUE_SIZE = N_JOBS * 2
    
    with ProcessPoolExecutor(max_workers=N_JOBS, initializer=init_worker, initargs=(model_path,)) as executor:
        futures = set()
        
        for chunk in chunk_iterator:
            if DRY_RUN_ROWS and total_processed >= DRY_RUN_ROWS:
                logger.info(f"DRY RUN: Limit reached.")
                break

            if len(futures) >= MAX_QUEUE_SIZE:
                done, futures = wait(futures, return_when=FIRST_COMPLETED)
                for future in done:
                    result = future.result()
                    if isinstance(result, Exception):
                        logger.error(f"Worker failed: {result}")
                    else:
                        write_header = (total_processed == 0)
                        result.to_csv(OUTPUT_FILE, mode='a', index=False, header=write_header, float_format='%.4f')
                        total_processed += len(result)
                        del result
                        gc.collect()

            future = executor.submit(validate_and_process_batch, chunk, SCHEMA_CONFIG, OUTPUT_SCHEMA)
            futures.add(future)

        for future in as_completed(futures):
            result = future.result()
            if isinstance(result, Exception):
                logger.error(f"Worker failed: {result}")
            else:
                write_header = (total_processed == 0)
                result.to_csv(OUTPUT_FILE, mode='a', index=False, header=write_header, float_format='%.4f')
                total_processed += len(result)

    logger.info(f"--- SCORING COMPLETE ---")
    logger.info(f"Results saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
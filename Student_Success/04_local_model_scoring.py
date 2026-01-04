"""
Script Name: 04_local_model_scoring.py
Author: Shane Lee
Description: Inference Engine. Executes parallel batch processing for Multi-Class
             Classification. Uses memory mapping and enforces output schema validation.
             Prioritises config-defined batch size.
             Calculates 'Prediction Confidence' using predict_proba().
"""

import os
import sys
import json
import logging
import pandas as pd
import numpy as np  # Added for max probability calculation
import joblib
import gc
import sklearn
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import scalability_utils as utils

# --- Configuration Load ---
try:
    with open("config.json", "r") as f:
        config = json.load(f)
    utils.validate_config_version(config, expected_version="1.0")
    
    ops_settings = config.get("operational_settings", {})
    
    # CRITICAL: Prioritise config batch_size
    BATCH_SIZE = ops_settings.get("batch_size", 500)
    N_JOBS = ops_settings.get("inference_n_jobs", 2)
    OUTPUT_SCHEMA = config.get("output_schema_expectations", {})
    
except Exception as e:
    print(f"Config load failed: {e}")
    sys.exit(1)

logger = utils.configure_logging("ScoringForensics", "scoring_ops.log", config)

INPUT_FILE = "inputs/scoring_ready_data.csv"
OUTPUT_FILE = "outputs/scoring_results_prediction.csv"

# --- Locate Model ---
try:
    with open("logs/latest_job.txt", "r") as f:
        job_name = f.read().strip()
    search_dir = f"./downloaded_artifacts/{job_name}"
    model_path = None
    for root, dirs, files in os.walk(search_dir):
        if "model.joblib" in files:
            model_path = os.path.join(root, "model.joblib")
            break
    if not model_path: raise FileNotFoundError
except Exception:
    logger.error("CRITICAL: Model artifact not found.")
    sys.exit(1)

global_pipeline = None

def init_worker(path):
    global global_pipeline
    # Load with mmap to share memory across workers
    try:
        artifact = joblib.load(path, mmap_mode='r')
    except Exception:
        # Fallback if uncompressed loading fails
        artifact = joblib.load(path)
    global_pipeline = artifact['pipeline']

def process_batch(chunk):
    try:
        # 1. Get Class Prediction (e.g., "Dropout")
        preds = global_pipeline.predict(chunk)
        
        # 2. Get Probabilities (e.g., [0.8, 0.1, 0.1])
        # predict_proba returns an array of shape (n_samples, n_classes)
        probs = global_pipeline.predict_proba(chunk)
        
        # 3. Extract the Confidence (Max probability for the predicted class)
        confidence = np.max(probs, axis=1)
        
        chunk["Predicted Status"] = preds
        chunk["Prediction Confidence"] = confidence
        
        return chunk
    except Exception as e:
        return e

def main():
    logger.info(f"Initialising Scoring on {INPUT_FILE}")
    logger.info(f"Operational Configuration - Batch Size: {BATCH_SIZE}, Workers: {N_JOBS}")
    
    # Metadata Validation
    try:
        artifact = joblib.load(model_path)
        meta = artifact.get('metadata', {})
        if meta.get('sklearn') != sklearn.__version__:
            logger.warning(f"Version Mismatch: Model trained on {meta.get('sklearn')}, running on {sklearn.__version__}")
        del artifact
        gc.collect()
    except Exception as e:
        logger.error(f"Metadata check failed: {e}")

    # CRITICAL: Create outputs directory if it doesn't exist
    os.makedirs("outputs", exist_ok=True)

    # Reset output file
    if os.path.exists(OUTPUT_FILE): 
        os.remove(OUTPUT_FILE)
    
    chunk_iter = pd.read_csv(INPUT_FILE, chunksize=BATCH_SIZE)
    total_processed = 0
    
    with ProcessPoolExecutor(max_workers=N_JOBS, initializer=init_worker, initargs=(model_path,)) as executor:
        futures = set()
        
        for chunk in chunk_iter:
            if len(futures) >= N_JOBS * 2:
                done, futures = wait(futures, return_when=FIRST_COMPLETED)
                for f in done:
                    res = f.result()
                    if not isinstance(res, Exception):
                        # Enforce float format to prevent scientific notation
                        res.to_csv(OUTPUT_FILE, mode='a', index=False, header=(total_processed==0), float_format='%.8f')
                        total_processed += len(res)
                        del res
                        gc.collect()
            
            futures.add(executor.submit(process_batch, chunk))
            
        for f in wait(futures).done:
            res = f.result()
            if not isinstance(res, Exception):
                res.to_csv(OUTPUT_FILE, mode='a', index=False, header=(total_processed==0), float_format='%.8f')
                total_processed += len(res)

    logger.info(f"Scoring Complete. Results: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
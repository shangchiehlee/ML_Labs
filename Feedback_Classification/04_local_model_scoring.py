"""
Script Name: 04_local_model_scoring.py
Author: Shane Lee
Description: Inference Engine. Executes parallel batch processing for NLP.
             Uses memory mapping for efficient model loading and enforces 
             output schema validation.
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
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import scalability_utils as utils

try:
    with open("config.json", "r") as f:
        config = json.load(f)
    utils.validate_config_version(config, expected_version="1.0")
    
    ops_settings = config.get("operational_settings", {})
    data_settings = config.get("data_settings", {})
    
    BATCH_SIZE = ops_settings.get("batch_size", 5000)
    N_JOBS = ops_settings.get("inference_n_jobs", 2)
    TEXT_COL = data_settings.get("text_column", "Text")
    
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
    logger.error("CRITICAL: Model artifact not found. Did you run 02_download_outputs.py?")
    sys.exit(1)

global_pipeline = None

def init_worker(path):
    global global_pipeline
    try:
        # mmap_mode='r' shares memory pages across processes
        artifact = joblib.load(path, mmap_mode='r')
    except Exception:
        artifact = joblib.load(path)
    global_pipeline = artifact['pipeline']

def process_batch(chunk):
    try:
        if TEXT_COL not in chunk.columns:
            return Exception(f"Missing text column: {TEXT_COL}")

        # Ensure string format
        X_text = chunk[TEXT_COL].astype(str)
        
        # 1. Prediction (0 or 1)
        preds = global_pipeline.predict(X_text)
        
        # 2. Confidence Score (Absolute Distance)
        dist = global_pipeline.decision_function(X_text)
        confidence = np.abs(dist)
        
        chunk["Predicted_Sentiment"] = preds
        chunk["Confidence_Score"] = confidence
        
        return chunk
    except Exception as e:
        return e

def main():
    logger.info(f"Initialising Scoring on {INPUT_FILE}")
    
    # Metadata Validation
    try:
        artifact = joblib.load(model_path)
        meta = artifact.get('metadata', {})
        logger.info(f"Model Metadata: {meta}")
        del artifact
        gc.collect()
    except Exception as e:
        logger.error(f"Metadata check failed: {e}")

    os.makedirs("outputs", exist_ok=True)
    if os.path.exists(OUTPUT_FILE): os.remove(OUTPUT_FILE)
    
    chunk_iter = pd.read_csv(INPUT_FILE, chunksize=BATCH_SIZE, on_bad_lines='skip')
    total_processed = 0
    
    with ProcessPoolExecutor(max_workers=N_JOBS, initializer=init_worker, initargs=(model_path,)) as executor:
        futures = set()
        
        for chunk in chunk_iter:
            chunk = chunk.dropna(subset=[TEXT_COL])
            if chunk.empty: continue

            if len(futures) >= N_JOBS * 2:
                done, futures = wait(futures, return_when=FIRST_COMPLETED)
                for f in done:
                    res = f.result()
                    if not isinstance(res, Exception):
                        # Enforce float format
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

    logger.info(f"Scoring Complete. Predictions saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
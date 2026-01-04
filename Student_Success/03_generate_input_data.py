"""
Script Name: 03_generate_input_data.py
Author: Shane Lee
Description: ETL Utility. Prepares data for scoring by cleaning headers and
             removing the 'Target' column (Blind Prediction).
             Prioritises config-defined batch size for chunking demonstration.
"""

import os
import sys
import json
import logging
import pandas as pd
import gc
from concurrent.futures import ThreadPoolExecutor
import scalability_utils as utils

# --- Configuration Load ---
try:
    with open("config.json", "r") as f:
        config = json.load(f)
    utils.validate_config_version(config, expected_version="1.0")
    
    ops_settings = config.get("operational_settings", {})
    
    # CRITICAL: Prioritise config batch_size over dynamic calculation
    # This ensures we chunk the 4425 rows into batches of 500
    BATCH_SIZE = ops_settings.get("batch_size", 500)
    DRY_RUN_ROWS = ops_settings.get("dry_run_rows", None)
    
except Exception as e:
    print(f"Config load failed: {e}")
    sys.exit(1)

logger = utils.configure_logging("DataPrepForensics", "data_prep_ops.log", config)

def save_chunk(chunk, path, is_first_chunk):
    mode = 'w' if is_first_chunk else 'a'
    chunk.to_csv(path, mode=mode, index=False, header=is_first_chunk)
    return len(chunk)

if __name__ == "__main__":
    # CRITICAL FIX: Point to the 'inputs' directory based on file structure
    RAW_INPUT = "inputs/scoring_input_data.csv"
    
    os.makedirs("inputs", exist_ok=True)
    OUTPUT_PATH = "inputs/scoring_ready_data.csv"
    DLQ_PATH = "inputs/dlq_failed_chunks.csv"
    TARGET_COL = "Target"

    if not os.path.exists(RAW_INPUT):
        logger.error(f"CRITICAL: Input file {RAW_INPUT} not found.")
        sys.exit(1)

    logger.info(f"Processing raw data from: {RAW_INPUT}")
    logger.info(f"Operational Configuration - Batch Size: {BATCH_SIZE}")
    
    chunk_counter = 0
    total_rows = 0
    
    # Async I/O Executor
    with ThreadPoolExecutor(max_workers=1) as executor:
        futures = []
        
        for chunk in pd.read_csv(RAW_INPUT, chunksize=BATCH_SIZE):
            chunk_counter += 1
            
            try:
                # 1. Clean columns: Remove tabs, spaces, quotes, AND apostrophes
                chunk.columns = [c.strip().replace('"', '').replace("'", "") for c in chunk.columns]
                
                # 2. Drop Target for Scoring (Simulate real prediction)
                if TARGET_COL in chunk.columns:
                    chunk.drop(columns=[TARGET_COL], inplace=True)
                
                # 3. Dry Run Check
                if DRY_RUN_ROWS and total_rows >= DRY_RUN_ROWS:
                    logger.info(f"DRY RUN: Limit of {DRY_RUN_ROWS} rows reached.")
                    break

                is_first = (chunk_counter == 1)
                future = executor.submit(save_chunk, chunk, OUTPUT_PATH, is_first)
                futures.append(future)
                total_rows += len(chunk)
                
                logger.info(f"Processed Batch {chunk_counter}: {len(chunk)} rows")
                
            except Exception as e:
                logger.error(f"DLQ EVENT: Batch {chunk_counter} failed. {e}")
                chunk.to_csv(DLQ_PATH, mode='a', index=False)
            
            del chunk
            gc.collect()

        # Wait for all writes to complete
        for f in futures:
            f.result()

    logger.info(f"ETL Complete. {total_rows} rows prepared in {OUTPUT_PATH}")
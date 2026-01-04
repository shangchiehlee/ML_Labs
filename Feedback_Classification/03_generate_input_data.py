"""
Script Name: 03_generate_input_data.py
Author: Shane Lee
Description: ETL Utility. Simulates 'New/Unseen' data for inference by dropping 
             the target column. Implements Dead Letter Queues (DLQ) and 
             Asynchronous I/O for high-throughput processing.
"""

import os
import sys
import json
import logging
import pandas as pd
import gc
from concurrent.futures import ThreadPoolExecutor
import scalability_utils as utils

try:
    with open("config.json", "r") as f:
        config = json.load(f)
    utils.validate_config_version(config, expected_version="1.0")
    
    ops_settings = config.get("operational_settings", {})
    data_settings = config.get("data_settings", {})
    
    BATCH_SIZE = ops_settings.get("batch_size", 5000)
    DRY_RUN_ROWS = ops_settings.get("dry_run_rows", None)
    # CRITICAL: This is the column we MUST drop to simulate unseen data
    LABEL_COL = data_settings.get("label_column", "Score")
    
except Exception as e:
    print(f"Config load failed: {e}")
    sys.exit(1)

logger = utils.configure_logging("DataPrepForensics", "data_prep_ops.log", config)

def save_chunk(chunk, path, is_first_chunk):
    mode = 'w' if is_first_chunk else 'a'
    chunk.to_csv(path, mode=mode, index=False, header=is_first_chunk)
    return len(chunk)

if __name__ == "__main__":
    # Input: The raw historical data
    RAW_INPUT = "inputs/feedback_stream_raw.csv"
    # Output: The "Blind" data for scoring
    OUTPUT_PATH = "inputs/scoring_ready_data.csv"
    DLQ_PATH = "inputs/dlq_failed_chunks.csv"
    
    os.makedirs("inputs", exist_ok=True)

    if not os.path.exists(RAW_INPUT):
        logger.error(f"CRITICAL: Input file {RAW_INPUT} not found.")
        sys.exit(1)

    logger.info(f"Generating Blind Inference Data from: {RAW_INPUT}")
    
    chunk_counter = 0
    total_rows = 0
    
    with ThreadPoolExecutor(max_workers=1) as executor:
        futures = []
        
        for chunk in pd.read_csv(RAW_INPUT, chunksize=BATCH_SIZE, on_bad_lines='skip'):
            chunk_counter += 1
            
            try:
                # CRITICAL STEP: Drop the Target Column
                # This ensures the scoring engine is truly predicting, not cheating.
                if LABEL_COL in chunk.columns:
                    chunk.drop(columns=[LABEL_COL], inplace=True)
                
                # Dry Run Check
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

        for f in futures:
            f.result()

    logger.info(f"ETL Complete. Blind dataset ready at: {OUTPUT_PATH}")
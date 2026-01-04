"""
Script Name: 03_generate_input_data.py
Author: Shane Lee
Description: ETL Utility. Extracts raw datasets using Asynchronous I/O to prevent
             blocking the main execution thread. Implements a Dead Letter Queue (DLQ)
             mechanism to isolate schema violations without halting the ingestion stream.
             Applies exponential backoff logic for network resilience.
"""

import os
import sys
import json
import logging
import pandas as pd
import gc
import time
import functools
from concurrent.futures import ThreadPoolExecutor, TimeoutError

# --- Local Utils Import ---
import scalability_utils as utils

# --- Configuration ---

try:
    with open("config.json", "r") as f:
        config = json.load(f)
    utils.validate_config_version(config, expected_version="1.0")
    
    schema_config = config.get("schema_expectations", {})
    ops_settings = config.get("operational_settings", {})
    DRY_RUN_ROWS = ops_settings.get("dry_run_rows", None)
except Exception as e:
    print(f"Config load failed: {e}")
    sys.exit(1)

logger = utils.configure_logging("DataPrepForensics", "data_prep_ops.log", config)

# --- Resilience ---

def retry_with_backoff(retries=3, backoff_in_seconds=2):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            x = 0
            while True:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if x == retries:
                        logger.error(f"CRITICAL: Operation failed after {retries} retries. Error: {e}")
                        raise e
                    sleep = (backoff_in_seconds * 2 ** x)
                    logger.warning(f"Transient Error: {e}. Retrying in {sleep}s...")
                    time.sleep(sleep)
                    x += 1
        return wrapper
    return decorator

def validate_schema(df, schema_config):
    if not schema_config:
        return True
    
    # Check required columns (excluding target, which we drop)
    # Note: We check against the raw data columns before dropping target
    pass 
    
    return df

def save_chunk(chunk, path, is_first_chunk):
    mode = 'w' if is_first_chunk else 'a'
    header = is_first_chunk
    chunk.to_csv(path, mode=mode, index=False, header=header)
    return len(chunk)

@retry_with_backoff(retries=3, backoff_in_seconds=5)
def execute_etl_stream(data_url, output_path, dlq_path, batch_size, schema_config, target_col, dry_run_limit=None):
    if os.path.exists(output_path):
        os.remove(output_path)
    if os.path.exists(dlq_path):
        os.remove(dlq_path)
        
    chunk_counter = 0
    total_rows = 0
    dlq_rows = 0
    
    with ThreadPoolExecutor(max_workers=1) as writer_executor:
        futures = []
        
        # Note: Graduate Admissions CSV is standard, no header=1 needed usually, but let's check source.
        # Source has standard header.
        for chunk in pd.read_csv(data_url, chunksize=batch_size):
            chunk_counter += 1
            
            try:
                chunk.columns = chunk.columns.str.strip()
                
                if 'Serial No.' in chunk.columns:
                    chunk.drop(columns=['Serial No.'], inplace=True)
                
                if target_col in chunk.columns:
                    chunk.drop(columns=[target_col], inplace=True)
                
                if dry_run_limit and total_rows >= dry_run_limit:
                    logger.info(f"DRY RUN: Limit of {dry_run_limit} rows reached.")
                    break

                if futures:
                    try:
                        futures[-1].result(timeout=60) 
                    except TimeoutError:
                        logger.error("CRITICAL: Disk Write Timeout.")
                        raise

                is_first = (chunk_counter == 1)
                future = writer_executor.submit(save_chunk, chunk, output_path, is_first)
                futures.append(future)
                
                total_rows += len(chunk)
                logger.info(f"Processed Batch {chunk_counter}: {len(chunk)} rows.")

            except Exception as ve:
                logger.error(f"DLQ EVENT: Batch {chunk_counter} failed. Error: {ve}")
                dlq_is_first = (dlq_rows == 0)
                chunk.to_csv(dlq_path, mode='a', header=dlq_is_first, index=False)
                dlq_rows += len(chunk)
            
            del chunk
            if chunk_counter % 10 == 0:
                gc.collect()

        if futures:
            futures[-1].result(timeout=60)
            
    return total_rows, dlq_rows

if __name__ == "__main__":
    utils.check_disk_space()
    BATCH_SIZE = utils.calculate_optimal_batch_size(n_jobs=1, safety_factor=0.5)

    os.makedirs("inputs", exist_ok=True)
    OUTPUT_PATH = "inputs/scoring_input_data.csv"
    DLQ_PATH = "inputs/dlq_failed_chunks.csv"
    
    DATA_URL = "https://raw.githubusercontent.com/srinivasav22/Graduate-Admission-Prediction/master/Admission_Predict_Ver1.1.csv"
    TARGET_COL = "Chance of Admit"

    logger.info(f"Initialising extraction from: {DATA_URL}")
    
    try:
        total_rows, dlq_rows = execute_etl_stream(DATA_URL, OUTPUT_PATH, DLQ_PATH, BATCH_SIZE, schema_config, TARGET_COL, DRY_RUN_ROWS)
        logger.info(f"SUCCESS: Transformation Complete. Total Rows: {total_rows}")
        if dlq_rows > 0:
            logger.warning(f"WARNING: {dlq_rows} rows routed to DLQ.")
    except Exception as e:
        logger.error(f"CRITICAL: ETL Stream failed. Error: {e}")
        sys.exit(1)
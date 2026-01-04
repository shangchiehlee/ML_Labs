"""
Script Name: 03_generate_input_data.py
Author: Shane Lee
Description: ETL Utility. Streams raw dataset from the source using asynchronous I/O.
             Implements a Dead Letter Queue (DLQ) to isolate schema violations without halting the ingestion pipeline.
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

# --- Configuration & Logging ---

try:
    with open("config.json", "r") as f:
        config = json.load(f)
    
    # Validate Schema Version
    utils.validate_config_version(config, expected_version="1.0")
    
    schema_config = config.get("schema_expectations", {})
    ops_settings = config.get("operational_settings", {})
    DRY_RUN_ROWS = ops_settings.get("dry_run_rows", None)
    # Prioritise config batch size, fallback to dynamic calculation
    CONFIG_BATCH_SIZE = ops_settings.get("batch_size", None)
except Exception as e:
    print(f"Config load failed: {e}")
    sys.exit(1)

logger = utils.configure_logging("DataPrepForensics", "data_prep_ops.log", config)

# --- Resilience Functions ---

def retry_with_backoff(retries=3, backoff_in_seconds=2):
    """
    Decorator to retry a function with exponential backoff.
    """
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

# --- ETL Functions ---

def validate_schema(df, schema_config):
    """
    Validates that the dataframe conforms to the contract in config.json.
    """
    if not schema_config:
        return True

    missing_cols = [col for col in schema_config.get("required_columns", []) if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Schema Violation: Missing columns {missing_cols}")
    
    type_expectations = schema_config.get("dtypes", {})
    for col, dtype in type_expectations.items():
        if col in df.columns:
            try:
                df[col] = df[col].astype(dtype)
            except Exception as e:
                logger.warning(f"Schema Warning: Column {col} could not be cast to {dtype}. Error: {e}")
    
    return df

def save_chunk(chunk, path, is_first_chunk):
    """
    Worker function for threaded file writing.
    """
    mode = 'w' if is_first_chunk else 'a'
    header = is_first_chunk
    chunk.to_csv(path, mode=mode, header=header)
    return len(chunk)

@retry_with_backoff(retries=3, backoff_in_seconds=5)
def execute_etl_stream(data_url, output_path, dlq_path, batch_size, schema_config, target_col, dry_run_limit=None):
    """
    Executes the download and transformation stream. 
    Wrapped in retry logic to handle network interruptions.
    Includes Dead Letter Queue (DLQ) logic for failed chunks.
    """
    # Ensure clean state for retry
    if os.path.exists(output_path):
        os.remove(output_path)
    if os.path.exists(dlq_path):
        os.remove(dlq_path)
        
    chunk_counter = 0
    total_rows = 0
    dlq_rows = 0
    
    # ThreadPool for writing to disk while main thread downloads next chunk
    with ThreadPoolExecutor(max_workers=1) as writer_executor:
        futures = []
        
        for chunk in pd.read_csv(data_url, header=1, index_col=0, chunksize=batch_size):
            chunk_counter += 1
            
            try:
                # 1. Transform
                if target_col in chunk.columns:
                    chunk.drop(columns=[target_col], inplace=True)
                
                # 2. Validate
                chunk = validate_schema(chunk, schema_config)
                
                # 3. Dry Run Check
                if dry_run_limit and total_rows >= dry_run_limit:
                    logger.info(f"DRY RUN: Limit of {dry_run_limit} rows reached. Halting extraction.")
                    break

                # 4. Async Write (Success Path)
                if futures:
                    try:
                        futures[-1].result(timeout=60) 
                    except TimeoutError:
                        logger.error("CRITICAL: Disk Write Timeout. Storage subsystem may be stalled.")
                        raise

                is_first = (chunk_counter == 1)
                future = writer_executor.submit(save_chunk, chunk, output_path, is_first)
                futures.append(future)
                
                rows_in_chunk = len(chunk)
                total_rows += rows_in_chunk
                logger.info(f"Processed Batch {chunk_counter}: {rows_in_chunk} rows.")

            except ValueError as ve:
                # DLQ Path: Schema Violation
                logger.error(f"DLQ EVENT: Batch {chunk_counter} failed validation. Routing to DLQ. Error: {ve}")
                dlq_is_first = (dlq_rows == 0)
                chunk.to_csv(dlq_path, mode='a', header=dlq_is_first)
                dlq_rows += len(chunk)
            
            # Explicit GC
            del chunk
            if chunk_counter % 10 == 0:
                gc.collect()

        # Ensure final chunk is written
        if futures:
            try:
                futures[-1].result(timeout=60)
            except TimeoutError:
                logger.error("CRITICAL: Final Disk Write Timeout.")
                raise
            
    return total_rows, dlq_rows

# --- Execution ---

if __name__ == "__main__":
    utils.check_disk_space()
    
    # Determine Batch Size: Config > Dynamic
    if CONFIG_BATCH_SIZE:
        BATCH_SIZE = CONFIG_BATCH_SIZE
        logger.info(f"Using Configured Batch Size: {BATCH_SIZE}")
    else:
        BATCH_SIZE = utils.calculate_optimal_batch_size(n_jobs=1, safety_factor=0.5)
        logger.info(f"Using Dynamic Batch Size: {BATCH_SIZE}")

    os.makedirs("inputs", exist_ok=True)
    OUTPUT_PATH = "inputs/scoring_input_data.csv"
    DLQ_PATH = "inputs/dlq_failed_chunks.csv"
    
    DATA_URL = "https://azuremlexamples.blob.core.windows.net/datasets/credit_card/default%20of%20credit%20card%20clients.csv"
    TARGET_COL = "default payment next month"

    logger.info(f"Initialising extraction from: {DATA_URL}")
    logger.info(f"Processing Mode: Async I/O (Chunk Size: {BATCH_SIZE})")
    if DRY_RUN_ROWS:
        logger.info(f"DRY RUN ENABLED: Processing limited to {DRY_RUN_ROWS} rows.")

    try:
        total_rows, dlq_rows = execute_etl_stream(DATA_URL, OUTPUT_PATH, DLQ_PATH, BATCH_SIZE, schema_config, TARGET_COL, DRY_RUN_ROWS)
        
        logger.info(f"SUCCESS: Transformation Complete.")
        logger.info(f"Total Rows Processed: {total_rows}")
        logger.info(f"Input data saved to: {OUTPUT_PATH}")
        
        if dlq_rows > 0:
            logger.warning(f"WARNING: {dlq_rows} rows routed to Dead Letter Queue: {DLQ_PATH}")

    except Exception as e:
        logger.error(f"CRITICAL: ETL Stream failed permanently. Error: {e}")
        sys.exit(1)
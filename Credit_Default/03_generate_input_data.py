"""ETL utility for streaming data transformation.

Purpose:
    Streams raw datasets in chunks and writes output for inference.
    Uses a single worker thread to serialise disk writes and captures failed
    chunks in a DLQ file.

Workflow:
    1. Reads raw CSV data in chunks.
    2. Drops label column for blind inference simulation.
    3. Validates required columns and attempts dtype casting.
    4. Writes processed chunks to `scoring_input_data.csv`.
    5. Routes failed chunks to `dlq_failed_chunks.csv`.

Author: Shane Lee
Licence: MIT
"""

import gc
import logging
import sys
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd

# --- Local Utils Import ---
import scalability_utils as utils

# --- Configuration & Logging ---

config: Dict[str, Any] = utils.load_config()
logger: logging.Logger = utils.configure_logging("DataPrepForensics", "data_prep_ops.log", config)

ops_settings: Dict[str, Any] = config.get("operational_settings", {})
etl_settings: Dict[str, Any] = config.get("etl_settings", {})

# Operational Constants
DRY_RUN_ROWS: Optional[int] = ops_settings.get("dry_run_rows")
DISK_WRITE_TIMEOUT: int = etl_settings.get("disk_write_timeout_seconds", 60)
ROW_SIZE_EST: int = etl_settings.get("row_size_estimate_bytes", 1000)

# Batch Size Resolution
BATCH_SIZE: int = utils.resolve_batch_size(
    config,
    n_jobs=1,
    row_size_estimate_bytes=ROW_SIZE_EST
)


def save_chunk(chunk: pd.DataFrame, path: Path, is_first_chunk: bool) -> int:
    """Worker function for threaded disk serialisation.

    Args:
        chunk: DataFrame batch to persist.
        path: Target file path.
        is_first_chunk: Flag to determine if header should be written.

    Returns:
        int: Number of records successfully persisted.
    """
    mode: str = 'w' if is_first_chunk else 'a'
    chunk.to_csv(path, mode=mode, index=True, header=is_first_chunk)
    return len(chunk)


@utils.retry_with_backoff(retries=3, backoff_in_seconds=5, logger=logger)
def execute_etl_stream(
    data_url: str,
    output_path: Path,
    dlq_path: Path,
    batch_size: int,
    schema_def: Dict[str, Any],
    target_col: str,
    dry_run_limit: Optional[int] = None
) -> Tuple[int, int]:
    """Streams data through the transformation context.

    Args:
        data_url: Source URI for raw ingestion.
        output_path: Destination for validated inference data.
        dlq_path: Destination for malformed record isolation.
        batch_size: Row count per iteration.
        schema_def: Mandatory schema contract.
        target_col: Label column to strip for blind prediction.
        dry_run_limit: Optional record cap for testing.

    Returns:
        Tuple[int, int]: (Total Processed, Total DLQ).

    Raises:
        Exception: If the input stream is interrupted.
    """
    # Purge stale artifacts
    if output_path.exists():
        output_path.unlink()
    if dlq_path.exists():
        dlq_path.unlink()

    processed_count: int = 0
    dlq_count: int = 0

    # Single-worker thread serialises writes to preserve ordering
    with ThreadPoolExecutor(max_workers=1) as executor:
        last_future: Optional[Future[int]] = None

        try:
            # Chunked reader keeps memory use tied to batch size and overhead
            with pd.read_csv(data_url, chunksize=batch_size, header=1, index_col=0) as reader:
                for idx, chunk in enumerate(reader):
                    try:
                        # 1. Label Stripping (Blind Prediction Enforcement)
                        if target_col in chunk.columns:
                            chunk.drop(columns=[target_col], inplace=True)

                        # 2. Contract Validation
                        chunk = utils.validate_dataframe_schema(
                            chunk, schema_def, logger=logger
                        )

                        # 3. Dry Run Constraint
                        if dry_run_limit and processed_count >= dry_run_limit:
                            logger.info(f"DRY RUN: Hitting {dry_run_limit} row limit. Halting stream.")
                            break

                        # Serialised write barrier (ordering preserved)
                        # We wait for the previous write to complete to ensure file integrity order
                        if last_future:
                            last_future.result(timeout=DISK_WRITE_TIMEOUT)

                        is_first: bool = (idx == 0)
                        last_future = executor.submit(
                            save_chunk, chunk, output_path, is_first
                        )

                        processed_count += len(chunk)
                        logger.info(f"Buffered Batch {idx + 1}: {len(chunk)} rows.")

                    except Exception as ve:
                        logger.error(f"DLQ EVENT: Batch {idx + 1} failed schema validation. Error: {ve}")
                        # Synchronous DLQ write to ensure capture integrity
                        dlq_header: bool = not dlq_path.exists()
                        chunk.to_csv(dlq_path, mode='a', index=False, header=dlq_header)
                        dlq_count += len(chunk)

                    # Explicit Garbage Collection
                    del chunk
                    if (idx + 1) % 10 == 0:
                        gc.collect()

            # Final Write Synchronization
            if last_future:
                last_future.result(timeout=DISK_WRITE_TIMEOUT)

        except Exception as e:
            logger.error(f"ETL STREAM FAILURE: {e}")
            raise

    return processed_count, dlq_count


def main() -> None:
    """Main ETL orchestration flow."""
    utils.check_disk_space()

    # Path Configuration
    utils.ProjectPaths.INPUTS.mkdir(exist_ok=True)
    output_path: Path = utils.ProjectPaths.SCORING_INPUT
    dlq_path: Path = utils.ProjectPaths.DLQ

    schema_expectations: Dict[str, Any] = config.get("schema_expectations", {})
    feat_defs: Dict[str, Any] = config.get("feature_definitions", {})
    target_col: str = feat_defs.get("target", "default payment next month")
    data_url: str = ops_settings.get("data_source_url", "")

    if not data_url:
        logger.error("CRITICAL: Data source URI omitted from configuration.")
        sys.exit(1)

    logger.info(f"Initialising stream from: {data_url}")

    try:
        total, dlq = execute_etl_stream(
            data_url,
            output_path,
            dlq_path,
            BATCH_SIZE,
            schema_expectations,
            target_col,
            DRY_RUN_ROWS
        )
        logger.info(f"SUCCESS: ETL Synchronised. Total: {total}, DLQ: {dlq}")
    except Exception as e:
        logger.error(f"CRITICAL: Pipeline terminal failure. Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

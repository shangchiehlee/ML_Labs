"""ETL utility for streaming student outcome data transformation.

Purpose:
    Prepares raw datasets for blind inference by stripping labels and normalising
    schema headers. Processes input in chunks to avoid full dataset materialisation.
    Implements a Dead Letter Queue (DLQ) mechanism to isolate chunks that raise
    processing exceptions, and writes output via a single-worker executor with
    explicit ordering.

Workflow:
    1. Reads raw CSV in chunks.
    2. Normalises headers (removes quotes, whitespace).
    3. Drops label column for blind inference simulation.
    4. Writes valid chunks to `scoring_input_data.csv`.
    5. Writes failed chunks to `dlq_failed_chunks.csv`.

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
DRY_RUN_ROWS: Optional[int] = ops_settings.get("dry_run_rows")

# Resource Constraints & Timeouts
ROW_SIZE_EST: int = ops_settings.get("row_size_estimate_bytes", 1024)
DISK_TIMEOUT: int = ops_settings.get("disk_write_timeout_seconds", 60)

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
    chunk.to_csv(path, mode=mode, index=False, header=is_first_chunk)
    return len(chunk)


@utils.retry_with_backoff(retries=3, backoff_in_seconds=5, logger=logger)
def execute_etl_stream(
    input_path: Path,
    output_path: Path,
    dlq_path: Path,
    batch_size: int,
    target_col: str,
    dry_run_limit: Optional[int] = None
) -> Tuple[int, int]:
    """Streams data through the transformation context.

    Args:
        input_path: Source path for raw ingestion.
        output_path: Destination for validated inference data.
        dlq_path: Destination for malformed record isolation.
        batch_size: Row count per iteration.
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

    with ThreadPoolExecutor(max_workers=1) as executor:
        last_future: Optional[Future[int]] = None

        try:
            # Chunked reader to bound memory to batch size
            with pd.read_csv(input_path, chunksize=batch_size) as reader:
                for idx, chunk in enumerate(reader):
                    try:
                        # 1. Header Normalisation
                        chunk.columns = [
                            str(c).strip().replace('"', '').replace("'", "")
                            for c in chunk.columns
                        ]

                        # 2. Label Stripping (Blind Prediction Enforcement)
                        if target_col in chunk.columns:
                            chunk.drop(columns=[target_col], inplace=True)

                        # 3. Dry Run Constraint
                        if dry_run_limit and processed_count >= dry_run_limit:
                            logger.info(f"DRY RUN: Hitting {dry_run_limit} row limit. Halting stream.")
                            break

                        # Write barrier to preserve ordering
                        if last_future:
                            last_future.result(timeout=DISK_TIMEOUT)

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
                last_future.result(timeout=DISK_TIMEOUT)

        except Exception as e:
            logger.error(f"ETL STREAM FAILURE: {e}")
            raise

    return processed_count, dlq_count


def main() -> None:
    """Main ETL orchestration flow."""
    utils.check_disk_space()
    utils.ProjectPaths.INPUTS.mkdir(parents=True, exist_ok=True)

    # Path Configuration
    data_dir: str = ops_settings.get("data_directory", "inputs")
    raw_input: Path = Path(data_dir) / "scoring_input_data.csv"
    output_path: Path = utils.ProjectPaths.SCORING_INPUT
    dlq_path: Path = utils.ProjectPaths.DLQ

    feature_defs: Dict[str, Any] = config.get("feature_definitions", {})
    target_col: str = feature_defs.get("target", "Target")

    if not raw_input.exists():
        logger.error(f"CRITICAL: Raw ingestion source {raw_input} missing.")
        sys.exit(1)
    if raw_input.resolve() == output_path.resolve():
        logger.error(
            "CRITICAL: Raw ingestion source resolves to the scoring output path. "
            "Configure 'data_directory' to a different location or supply a distinct raw input file."
        )
        sys.exit(1)

    logger.info(f"Initialising stream from: {raw_input}")
    logger.info(f"Operational Constraints: Batch={BATCH_SIZE}")

    try:
        total, dlq = execute_etl_stream(
            raw_input,
            output_path,
            dlq_path,
            BATCH_SIZE,
            target_col,
            DRY_RUN_ROWS
        )
        logger.info(f"SUCCESS: ETL Synchronised. Total: {total}, DLQ: {dlq}")
    except Exception as e:
        logger.error(f"CRITICAL: Pipeline terminal failure. Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

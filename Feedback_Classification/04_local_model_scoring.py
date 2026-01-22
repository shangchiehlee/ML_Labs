"""Inference engine for parallel batch NLP scoring.

Purpose:
    Loads model artefacts with joblib mmap_mode='r' during worker initialisation to
    request memory-mapped reads where supported. Streams input data in chunks, scores
    in parallel workers, and writes predictions plus confidence scores to CSV, with
    schema validation when configured.

Workflow:
    1. Loads the model artefact in each worker using joblib with mmap_mode='r'.
    2. Spawns parallel workers using `ProcessPoolExecutor`.
    3. Streams input data in chunks.
    4. Workers compute predictions and confidence scores.
    5. Aggregator writes results to CSV.

Author: Shane Lee
Licence: MIT
"""

import gc
import logging
import sys
from concurrent.futures import (
    FIRST_COMPLETED,
    ProcessPoolExecutor,
    as_completed,
    wait,
)
from pathlib import Path
from typing import Any, Dict, Optional, Set, Union

import joblib
import numpy as np
import pandas as pd
import sklearn

# --- Local Utils Import ---
import scalability_utils as utils

# --- Configuration & Logging ---

config: Dict[str, Any] = utils.load_config()
logger: logging.Logger = utils.configure_logging("ScoringForensics", "scoring_ops.log", config)

inf_settings: Dict[str, Any] = config.get("inference_settings", {})
feat_defs: Dict[str, Any] = config.get("feature_definitions", {})
out_schema: Dict[str, Any] = config.get("output_schema_expectations", {})

# Operational Constants
N_JOBS: int = inf_settings.get("n_jobs", 2)
DRY_RUN_ROWS: Optional[int] = config.get("operational_settings", {}).get("dry_run_rows")
ROW_SIZE_EST: int = 2048

# Batch Size Resolution
BATCH_SIZE: int = utils.resolve_batch_size(
    config,
    n_jobs=N_JOBS,
    row_size_estimate_bytes=ROW_SIZE_EST
)

# Shared Worker State
# Global variable initialised in worker processes
global_pipeline: Optional[Any] = None


def init_worker(model_path: Path) -> None:
    """Initialises a worker process by loading the model.

    Uses joblib.load with mmap_mode='r' to request memory-mapped reads where
    supported.

    Args:
        model_path: Path to the serialised joblib artefact.
    """
    global global_pipeline
    try:
        # mmap_mode='r' requests read-only memory-mapped access
        artifact: Any = joblib.load(model_path, mmap_mode='r')
        global_pipeline = artifact['pipeline'] if isinstance(artifact, dict) else artifact
    except Exception as e:
        # Stdout/Stderr may not be captured by logger in subprocess
        sys.stderr.write(f"Inference worker failed to spawn: {e}\n")


def score_batch(
    chunk: pd.DataFrame,
    text_col: str,
    schema_def: Dict[str, Any]
) -> Union[pd.DataFrame, Exception]:
    """Inference logic executed within each parallel worker process.

    Args:
        chunk: Data batch to classify.
        text_col: Key for unstructured text.
        schema_def: Schema contract for output validation.

    Returns:
        Union[pd.DataFrame, Exception]: Scored batch or failure reason.
    """
    try:
        if global_pipeline is None:
            raise RuntimeError("Pipeline missing in worker context.")

        texts: pd.Series = chunk[text_col].astype(str)
        chunk["Predicted_Sentiment"] = global_pipeline.predict(texts)

        # Deriving Confidence (Decision Function Distance or Proba Max)
        if hasattr(global_pipeline, "decision_function"):
            # Distance from hyperplane
            chunk["Confidence_Score"] = np.abs(global_pipeline.decision_function(texts))
        elif hasattr(global_pipeline, "predict_proba"):
            # Max probability class
            chunk["Confidence_Score"] = np.max(global_pipeline.predict_proba(texts), axis=1)
        else:
            chunk["Confidence_Score"] = 0.0

        # Contract Validation
        utils.validate_dataframe_schema(chunk, schema_def)

        return chunk
    except Exception as e:
        return e


def main() -> None:
    """Main inference orchestration flow."""
    # 1. Identify Artifact
    job_name: Optional[str] = utils.get_latest_job_id()
    if not job_name:
        logger.error("CRITICAL: No active job identifier found in local state.")
        sys.exit(1)

    model_path: Optional[Path] = utils.find_model_artifact(utils.ProjectPaths.ARTIFACTS / job_name)
    if not model_path:
        logger.error(f"CRITICAL: model.joblib missing in artifacts for job {job_name}.")
        sys.exit(1)

    logger.info(f"Initialising Scoring context [Job: {job_name}]")

    # 2. Version Audit (Main Process)
    try:
        artifact: Any = joblib.load(model_path)
        train_ver: str = artifact.get("metadata", {}).get("sklearn", "Unknown")
        if train_ver != sklearn.__version__:
            logger.warning(
                f"VERSION SKEW: Model trained on Sklearn {train_ver}, "
                f"runtime is {sklearn.__version__}."
            )
        del artifact
        gc.collect()
    except Exception as e:
        logger.error(f"Integrity check failed: {e}")
        sys.exit(1)

    # 3. Stream Setup
    input_file: Path = utils.ProjectPaths.SCORING_INPUT
    output_file: Path = utils.ProjectPaths.SCORING_OUTPUT
    text_col: str = feat_defs.get("text_column", "Text")

    if not input_file.exists():
        logger.error(f"Input data missing: {input_file}")
        sys.exit(1)

    utils.ProjectPaths.OUTPUTS.mkdir(exist_ok=True)
    if output_file.exists():
        output_file.unlink()

    total_count: int = 0
    
    # Use context manager for reader
    with pd.read_csv(input_file, chunksize=BATCH_SIZE, on_bad_lines='skip') as reader:
        # 4. Parallel Execution Loop
        with ProcessPoolExecutor(
            max_workers=N_JOBS,
            initializer=init_worker,
            initargs=(model_path,)
        ) as executor:
            futures: Set[Any] = set()

            for chunk in reader:
                if DRY_RUN_ROWS and total_count >= DRY_RUN_ROWS:
                    break

                # Backpressure Control
                if len(futures) >= N_JOBS * 2:
                    done, futures = wait(futures, return_when=FIRST_COMPLETED)
                    for f in done:
                        res: Union[pd.DataFrame, Exception] = f.result()
                        if not isinstance(res, Exception):
                            res.to_csv(
                                output_file,
                                mode='a',
                                index=False,
                                header=(total_count == 0),
                                float_format='%.8f'
                            )
                            total_count += len(res)
                            logger.info(f"Processed: {total_count} samples.")
                        else:
                            logger.error(f"Worker process crashed: {res}")

                futures.add(executor.submit(score_batch, chunk, text_col, out_schema))

            # Drain completion queue
            for f in as_completed(futures):
                res = f.result()
                if not isinstance(res, Exception):
                    res.to_csv(
                        output_file,
                        mode='a',
                        index=False,
                        header=(total_count == 0),
                        float_format='%.8f'
                    )
                    total_count += len(res)

    logger.info(f"--- SCORING COMPLETE [Total: {total_count}] ---")


if __name__ == "__main__":
    main()

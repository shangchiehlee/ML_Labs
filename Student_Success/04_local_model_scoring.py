"""Inference engine for parallel batch student outcome scoring.

Purpose:
    Loads the model with joblib using mmap_mode="r" during worker initialisation and
    streams input data in chunks for batch scoring. Validates output schema when
    expectations are configured and calculates confidence scores for multi-class
    predictions.

Workflow:
    1. Loads the model artefact using `joblib` with mmap mode.
    2. Spawns parallel workers using `ProcessPoolExecutor`.
    3. Streams input data in chunks.
    4. Workers compute multi-class predictions and confidence.
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

ops_settings: Dict[str, Any] = config.get("operational_settings", {})
out_schema: Dict[str, Any] = config.get("output_schema_expectations", {})

# Operational Constants
N_JOBS: int = ops_settings.get("inference_n_jobs", 2)
ROW_SIZE_EST: int = ops_settings.get("row_size_estimate_bytes", 1024)
DRY_RUN_ROWS: Optional[int] = ops_settings.get("dry_run_rows")
FLOAT_FORMAT: str = ops_settings.get("scoring_float_format", "%.8f")

# Batch Size Resolution
BATCH_SIZE: int = utils.resolve_batch_size(
    config, 
    n_jobs=N_JOBS, 
    row_size_estimate_bytes=ROW_SIZE_EST
)

# Shared Worker State
# Global variable initialised by worker processes
global_pipeline: Optional[Any] = None


def init_worker(model_path: Path) -> None:
    """Initialises worker process by loading the model into shared memory.

    Loads the model artifact with joblib mmap_mode="r" to enable memory-mapped
    reads where supported.

    Args:
        model_path: Path to the serialised joblib artifact.
    """
    global global_pipeline
    try:
        # mmap_mode='r' enables read-only kernel page sharing
        artifact: Any = joblib.load(model_path, mmap_mode="r")
        # Support both raw pipeline and dict-wrapped pipeline
        global_pipeline = artifact.get("pipeline") if isinstance(artifact, dict) else artifact
    except Exception as e:
        sys.stderr.write(f"Inference worker failed to spawn: {e}\n")


def score_batch(
    chunk: pd.DataFrame, 
    output_schema_def: Dict[str, Any]
) -> Union[pd.DataFrame, Exception]:
    """Inference logic executed within each parallel worker process.

    Args:
        chunk: Data batch to classify.
        output_schema_def: Schema contract for validation.

    Returns:
        Union[pd.DataFrame, Exception]: Scored batch or failure reason.
    """
    try:
        if global_pipeline is None:
            raise RuntimeError("Pipeline missing in worker context.")

        # 1. Multi-Class Prediction
        preds: np.ndarray = global_pipeline.predict(chunk)

        # 2. Confidence Estimation (Probability Max)
        if hasattr(global_pipeline, "predict_proba"):
            probs: np.ndarray = global_pipeline.predict_proba(chunk)
            confidence: np.ndarray = np.max(probs, axis=1)
        else:
            confidence = np.zeros(len(preds))

        chunk["Predicted Status"] = preds
        chunk["Prediction Confidence"] = confidence

        # 3. Contract Validation
        utils.validate_dataframe_schema(chunk, output_schema_def)

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
        metadata: Dict[str, Any] = artifact.get("metadata", {})
        train_ver: str = metadata.get("sklearn", "Unknown")
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

    if not input_file.exists():
        logger.error(f"Input data missing: {input_file}")
        sys.exit(1)

    utils.ProjectPaths.OUTPUTS.mkdir(exist_ok=True)
    if output_file.exists():
        output_file.unlink()

    total_count: int = 0
    
    # 4. Parallel Execution Loop
    try:
        # Use context manager to ensure file handle is closed
        with pd.read_csv(input_file, chunksize=BATCH_SIZE) as reader:
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
                                    mode="a", 
                                    index=False, 
                                    header=(total_count == 0), 
                                    float_format=FLOAT_FORMAT
                                )
                                total_count += len(res)
                                logger.info(f"Processed: {total_count} records.")
                            else:
                                logger.error(f"Worker process crashed: {res}")

                    futures.add(executor.submit(score_batch, chunk, out_schema))

                # Drain completion queue
                for f in as_completed(futures):
                    res = f.result()
                    if not isinstance(res, Exception):
                        res.to_csv(
                            output_file, 
                            mode="a", 
                            index=False, 
                            header=(total_count == 0), 
                            float_format=FLOAT_FORMAT
                        )
                        total_count += len(res)

    except Exception as e:
        logger.error(f"CRITICAL: Scoring engine failure: {e}")
        sys.exit(1)

    logger.info(f"--- SCORING COMPLETE [Total: {total_count}] ---")


if __name__ == "__main__":
    main()

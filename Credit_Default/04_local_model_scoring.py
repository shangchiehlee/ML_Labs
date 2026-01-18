"""Inference engine for parallel batch credit risk scoring.

Purpose:
    Uses joblib.load(..., mmap_mode="r") during worker initialisation to request
    memory-mapped reads where supported. Streams input data in chunks, scores in
    parallel workers, and validates output against the configured schema.

Workflow:
    1. Loads the model artefact using `joblib` with mmap mode.
    2. Spawns parallel workers using `ProcessPoolExecutor`.
    3. Streams input data in chunks.
    4. Workers compute probabilities.
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
from typing import Any, Dict, List, Optional, Set, Union

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
inference_settings: Dict[str, Any] = config.get("inference_settings", {})
schema_config: Dict[str, Any] = config.get("schema_expectations", {})
out_schema: Dict[str, Any] = config.get("output_schema_expectations", {})

# Operational Constants
N_JOBS: int = inference_settings.get("inference_n_jobs", 2)
ROW_SIZE_EST: int = inference_settings.get("row_size_estimate_bytes", 1500)
DRY_RUN_ROWS: Optional[int] = ops_settings.get("dry_run_rows")

# Batch Size Resolution
BATCH_SIZE: int = utils.resolve_batch_size(
    config,
    n_jobs=N_JOBS,
    row_size_estimate_bytes=ROW_SIZE_EST
)

# --- Shared Worker State ---
# Global variables hold per-process model state initialised by init_worker
global_preprocessor: Optional[Any] = None
global_ensemble_models: Optional[List[Any]] = None
global_single_model: Optional[Any] = None
is_ensemble_mode: bool = False


def init_worker(model_path: Path) -> None:
    """Initialises worker process by loading the model into memory.

    Uses joblib.load with mmap_mode="r" to request memory-mapped reads where
    supported.

    Args:
        model_path: Path to the serialised joblib artifact.
    """
    global global_preprocessor, global_ensemble_models, global_single_model, is_ensemble_mode
    try:
        # mmap_mode='r' enables read-only memory mapping of the model file
        artifact: Any = joblib.load(model_path, mmap_mode="r")

        if isinstance(artifact, dict) and "ensemble_models" in artifact:
            is_ensemble_mode = True
            global_preprocessor = artifact["preprocessor"]
            global_ensemble_models = artifact["ensemble_models"]
        elif isinstance(artifact, dict) and "model" in artifact:
            is_ensemble_mode = False
            global_single_model = artifact["model"]
        else:
            is_ensemble_mode = False
            global_single_model = artifact

    except Exception as e:
        # We print to stderr as the logger might not be process-safe without configuration
        sys.stderr.write(f"Inference worker failed to spawn: {e}\n")


def score_batch(
    chunk: pd.DataFrame,
    input_schema: Dict[str, Any],
    output_schema_def: Dict[str, Any]
) -> Union[pd.DataFrame, Exception]:
    """Inference logic executed within each parallel worker process.

    Args:
        chunk: Data batch to classify.
        input_schema: Schema contract for input validation.
        output_schema_def: Schema contract for output validation.

    Returns:
        Union[pd.DataFrame, Exception]: Scored batch or failure reason.
    """
    try:
        # 1. Input Validation
        if input_schema:
            required: List[str] = input_schema.get("required_columns", [])
            # Allow target column to be missing (blind inference)
            missing: List[str] = [
                c for c in required
                if c not in chunk.columns and c != "default payment next month"
            ]
            if missing:
                raise ValueError(f"Input Schema Violation: Missing {missing}")

        # 2. Ensemble Inference (Soft Voting)
        if is_ensemble_mode and global_ensemble_models and global_preprocessor:
            x_trans: np.ndarray = global_preprocessor.transform(chunk)
            # Collect probabilities from all estimators
            probas_stack: List[np.ndarray] = [
                m.predict_proba(x_trans)[:, 1] for m in global_ensemble_models
            ]
            # Average probabilities
            chunk["Probability of Default"] = np.mean(probas_stack, axis=0)

        elif global_single_model:
            predictions: np.ndarray = global_single_model.predict_proba(chunk)
            chunk["Probability of Default"] = predictions[:, 1]
        else:
            raise RuntimeError("Pipeline missing in worker context.")

        # 3. Output Validation
        utils.validate_dataframe_schema(chunk, output_schema_def)

        return chunk
    except Exception as e:
        return e


def main() -> None:
    """Main inference orchestration flow."""
    # 1. Identify Artifact
    job_name: Optional[str] = utils.get_latest_job_id()
    if not job_name:
        logger.error("CRITICAL: No active job identified in local state.")
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
    
    # We use a context manager for the reader to ensure file handle closure
    with pd.read_csv(input_file, index_col=0, chunksize=BATCH_SIZE) as reader:
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

                # Backpressure Control: Reduce memory pressure by limiting active futures
                if len(futures) >= N_JOBS * 2:
                    done, futures = wait(futures, return_when=FIRST_COMPLETED)
                    for f in done:
                        res: Union[pd.DataFrame, Exception] = f.result()
                        if not isinstance(res, Exception):
                            res.to_csv(
                                output_file,
                                mode="a",
                                index=True,
                                header=(total_count == 0),
                                float_format="%.8f"
                            )
                            total_count += len(res)
                            logger.info(f"Processed: {total_count} records.")
                        else:
                            logger.error(f"Worker process crashed: {res}")

                futures.add(executor.submit(score_batch, chunk, schema_config, out_schema))

            # Drain completion queue
            for f in as_completed(futures):
                res = f.result()
                if not isinstance(res, Exception):
                    res.to_csv(
                        output_file,
                        mode="a",
                        index=True,
                        header=(total_count == 0),
                        float_format="%.8f"
                    )
                    total_count += len(res)
                else:
                    logger.error(f"Worker process crashed: {res}")

    logger.info(f"--- SCORING COMPLETE [Total: {total_count}] ---")


if __name__ == "__main__":
    main()

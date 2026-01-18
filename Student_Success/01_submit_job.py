"""Orchestration utility for Azure Machine Learning workload submission.

Purpose:
    Implements a programmatic build step to package shared utility libraries
    into the source distribution so the training script can import a local copy
    during cloud execution. Enforces execution timeout via CommandJobLimits.

Workflow:
    1. Loads local configuration and initialises the Azure ML client.
    2. Packages `scalability_utils.py` into the `src` build context.
    3. Defines the Azure ML `Job` specification with CLI arguments.
    4. Submits the workload and logs the Studio monitoring URL.

Author: Shane Lee
Licence: MIT
"""

import datetime
import json
import logging
import shutil
import sys
from pathlib import Path
from typing import Any, Dict

from azure.ai.ml import Input, command
from azure.ai.ml.entities import CommandJobLimits, Job

# --- Local Utils Import ---
import scalability_utils as utils

# --- Configuration & Logging ---

config: Dict[str, Any] = utils.load_config()
logger: logging.Logger = utils.configure_logging("JobSubmission", "submission_ops.log", config)


def package_shared_utilities(
    source_path: Path = Path("scalability_utils.py"),
    dest_dir: Path = Path("src")
) -> None:
    """Packages shared utility libraries into the source distribution folder.

    Copies the shared utility file into the build context so it is available
    for imports during job execution.

    Args:
        source_path: Path to the source utility file.
        dest_dir: Destination directory for the build context.

    Raises:
        SystemExit: If a critical utility file is missing.
    """
    logger.info("Packaging shared libraries for Cloud deployment...")

    # Resolve paths relative to the current script
    base_path: Path = Path(__file__).resolve().parent
    source_resolved: Path = (
        base_path / source_path if not source_path.is_absolute() else source_path
    )
    dest_resolved: Path = (
        base_path / dest_dir if not dest_dir.is_absolute() else dest_dir
    )
    dest_util: Path = dest_resolved / source_path.name

    if source_resolved.exists():
        dest_resolved.mkdir(parents=True, exist_ok=True)
        shutil.copy(source_resolved, dest_util)
        logger.info(f"Build Success: Injected '{source_path.name}' into '{dest_dir}'.")
    else:
        logger.error(f"CRITICAL: Mandatory utility '{source_path}' missing. Build aborted.")
        sys.exit(1)


def submit_job() -> None:
    """Main execution flow for Azure ML job orchestration.

    Loads configuration values, prepares the build context, and submits the
    training job to Azure ML.
    """
    logger.info("Initialising environment configuration...")

    ops_settings: Dict[str, Any] = config.get("operational_settings", {})
    hyperparams: Dict[str, Any] = config.get("training_hyperparameters", {})
    feature_defs: Dict[str, Any] = config.get("feature_definitions", {})

    # Operational Constants from Configuration
    TIMEOUT_SECONDS: int = ops_settings.get("training_timeout_seconds", 86400)
    BATCH_SIZE: int = ops_settings.get("batch_size", 500)
    DATA_DIR: str = ops_settings.get("data_directory", "inputs")
    ENV_URI: str = ops_settings.get(
        "azureml_environment",
        "azureml://registries/azureml/environments/sklearn-1.5/labels/latest"
    )
    EXP_NAME: str = ops_settings.get("experiment_name", "train_model_student_success")
    MODEL_NAME: str = ops_settings.get("registered_model_name", "student_success_model")

    # Input Path Resolution
    input_path: Path = Path(DATA_DIR) / "scoring_input_data.csv"

    if not input_path.exists():
        # Fallback to scoring_ready_data.csv if primary input is missing (legacy compat)
        alt_path: Path = Path(DATA_DIR) / "scoring_ready_data.csv"
        if alt_path.exists():
            logger.warning(
                f"Primary input '{input_path}' missing. Utilising processed artifact: '{alt_path}'"
            )
            input_path = alt_path
        else:
            logger.error(f"CRITICAL: Training data stream '{input_path}' not detected.")
            sys.exit(1)

    # Azure Connection Factory
    try:
        ml_client: Any = utils.get_azure_client(config)
        logger.info(f"Connection Established: Workspace '{ml_client.workspace_name}'")
    except Exception as e:
        logger.error(f"CRITICAL: Failed to connect to Azure ML Workspace: {e}")
        sys.exit(1)

    # Prepare Build Context
    package_shared_utilities()

    # Job Metadata Generation
    timestamp: str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    base_name: str = config.get("job_base_name", "student_success_train")
    job_name: str = f"{base_name}-{timestamp}"

    # CLI Transport Serialisation
    features_json_str: str = json.dumps(feature_defs)

    job_inputs: Dict[str, Any] = {
        "data": Input(type="uri_file", path=str(input_path)),
        "test_train_ratio": hyperparams.get("test_train_ratio", 0.2),
        "learning_rate": hyperparams.get("learning_rate", 0.01),
        "max_iter": hyperparams.get("max_iter", 50),
        "batch_size": BATCH_SIZE,
        "features_json": features_json_str,
        "registered_model_name": MODEL_NAME,
    }

    command_str: str = (
        "python main.py "
        "--data ${{inputs.data}} "
        "--test_train_ratio ${{inputs.test_train_ratio}} "
        "--learning_rate ${{inputs.learning_rate}} "
        "--max_iter ${{inputs.max_iter}} "
        "--batch_size ${{inputs.batch_size}} "
        "--features_json '${{inputs.features_json}}' "
        "--registered_model_name ${{inputs.registered_model_name}}"
    )

    # Define Cloud Workload
    job: Job = command(
        name=job_name,
        inputs=job_inputs,
        code=str(Path("src")),
        command=command_str,
        environment=ENV_URI,
        compute=config["compute_name"],
        experiment_name=EXP_NAME,
        display_name="student_success_prediction",
        limits=CommandJobLimits(timeout=TIMEOUT_SECONDS)
    )

    # Submission Logic
    logger.info(f"Submitting workload to Azure Compute (Timeout: {TIMEOUT_SECONDS}s)...")
    try:
        returned_job: Job = ml_client.create_or_update(job)
        utils.save_job_id(job_name)
        logger.info(f"Submission Complete. Monitoring URL: {returned_job.studio_url}")
    except Exception as e:
        logger.error(f"CRITICAL: Dispatch failure. Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    submit_job()

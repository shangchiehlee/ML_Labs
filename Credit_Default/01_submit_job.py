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
        source_path: Path to the source utility file (relative or absolute).
        dest_dir: Destination directory for the build context.

    Raises:
        SystemExit: If the critical utility file is missing.
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

    Negotiates environment configuration, prepares the build distribution,
    and dispatches the incremental training workload to Azure Compute Clusters.
    """
    logger.info("Initialising environment configuration...")

    # Extract settings with safe defaults
    ops_settings: Dict[str, Any] = config.get("operational_settings", {})
    hyperparams: Dict[str, Any] = config.get("training_hyperparameters", {})
    feature_defs: Dict[str, Any] = config.get("feature_definitions", {})

    # Operational Constants
    TIMEOUT_SECONDS: int = ops_settings.get("training_timeout_seconds", 86400)
    BATCH_SIZE: int = ops_settings.get("batch_size", 50000)

    # Feature Serialisation for CLI Transport
    # We pass complex structures as JSON strings to avoid command-line parsing issues
    features_json_str: str = json.dumps(feature_defs)

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
    base_name: str = config.get("job_base_name", "credit_default_train")
    job_name: str = f"{base_name}-{timestamp}"

    data_url: str = ops_settings.get(
        "data_source_url",
        "https://azuremlexamples.blob.core.windows.net/datasets/credit_card/"
        "default%20of%20credit%20card%20clients.csv"
    )

    # CLI Transport Configuration
    # These inputs are mapped to arguments in src/main.py
    job_inputs: Dict[str, Any] = {
        "data": Input(type="uri_file", path=data_url),
        "test_train_ratio": hyperparams.get("test_train_ratio", 0.2),
        "learning_rate": hyperparams.get("learning_rate", 0.1),
        "max_iter": hyperparams.get("max_iter", 100),
        "batch_size": BATCH_SIZE,
        "n_estimators": hyperparams.get("n_estimators", 7),
        "nystroem_components": hyperparams.get("nystroem_components", 1200),
        "features_json": features_json_str,
        "registered_model_name": "credit_defaults_model",
    }

    # Construct the execution command
    # Note: Strings inside ${{ }} are strict Azure ML syntax
    command_str: str = (
        "python main.py "
        "--data ${{inputs.data}} "
        "--test_train_ratio ${{inputs.test_train_ratio}} "
        "--learning_rate ${{inputs.learning_rate}} "
        "--max_iter ${{inputs.max_iter}} "
        "--batch_size ${{inputs.batch_size}} "
        "--n_estimators ${{inputs.n_estimators}} "
        "--nystroem_components ${{inputs.nystroem_components}} "
        "--features_json '${{inputs.features_json}}' "
        "--registered_model_name ${{inputs.registered_model_name}}"
    )

    # Define Cloud Workload
    job: Job = command(
        name=job_name,
        inputs=job_inputs,
        code=str(Path("src")),
        command=command_str,
        environment="azureml://registries/azureml/environments/sklearn-1.5/labels/latest",
        compute=config["compute_name"],
        experiment_name="train_model_credit_default",
        display_name="credit_default_prediction",
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

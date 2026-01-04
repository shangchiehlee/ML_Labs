"""
Script Name: 01_submit_job.py
Author: Shane Lee
Description: Orchestrates submission to Azure Compute.
             Includes Build Step to package shared utilities and passes
             batch_size configuration to the cloud environment.
"""

import json
import datetime
import os
import sys
import shutil
from azure.ai.ml import MLClient, command, Input
from azure.ai.ml.entities import CommandJobLimits
from azure.identity import DefaultAzureCredential
import scalability_utils as utils

# --- Configuration Load ---
try:
    with open("config.json", "r") as f:
        config = json.load(f)
    utils.validate_config_version(config, expected_version="1.0")
except Exception as e:
    print(f"CRITICAL: Configuration error: {e}")
    sys.exit(1)

logger = utils.configure_logging("JobSubmission", "submission_ops.log", config)
logger.info("Initialising environment configuration...")

ops_settings = config.get("operational_settings", {})
hyperparams = config.get("training_hyperparameters", {})
feature_defs = config.get("feature_definitions", {})

TIMEOUT_SECONDS = ops_settings.get("training_timeout_seconds", 86400)
# CRITICAL: Extract batch_size to pass to cloud
BATCH_SIZE = ops_settings.get("batch_size", 500)
features_json_str = json.dumps(feature_defs)

credential = DefaultAzureCredential()
ml_client = MLClient(
    credential=credential,
    subscription_id=config["subscription_id"],
    resource_group_name=config["resource_group"],
    workspace_name=config["workspace_name"],
)

# --- Build Step: Package Shared Utilities ---
logger.info("Packaging shared libraries for Cloud deployment...")
source_util = "scalability_utils.py"
dest_util = "src/scalability_utils.py"

if os.path.exists(source_util):
    os.makedirs("src", exist_ok=True)
    shutil.copy(source_util, dest_util)
    logger.info(f"Success: Copied {source_util} to {dest_util}")
else:
    logger.error(f"CRITICAL: Shared library {source_util} missing. Cannot proceed.")
    sys.exit(1)

# --- Job Definition ---
timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
job_name = f"{config['job_base_name']}-{timestamp}"

# CRITICAL FIX: Point to the 'inputs' directory based on file structure
LOCAL_DATA_PATH = "inputs/scoring_input_data.csv"

if not os.path.exists(LOCAL_DATA_PATH):
    logger.error(f"CRITICAL: Training data {LOCAL_DATA_PATH} not found.")
    sys.exit(1)

job_inputs = dict(
    data=Input(type="uri_file", path=LOCAL_DATA_PATH),
    test_train_ratio=hyperparams.get("test_train_ratio", 0.2),
    learning_rate=hyperparams.get("learning_rate", 0.01),
    max_iter=hyperparams.get("max_iter", 50),
    batch_size=BATCH_SIZE, # Pass batch size as input
    features_json=features_json_str,
    registered_model_name="student_success_model",
)

job_limits = CommandJobLimits(timeout=TIMEOUT_SECONDS)

# CRITICAL: Update command to accept --batch_size argument
job = command(
    name=job_name,
    inputs=job_inputs,
    code="./src",
    command="python main.py --data ${{inputs.data}} --test_train_ratio ${{inputs.test_train_ratio}} --learning_rate ${{inputs.learning_rate}} --max_iter ${{inputs.max_iter}} --batch_size ${{inputs.batch_size}} --features_json '${{inputs.features_json}}' --registered_model_name ${{inputs.registered_model_name}}",
    environment="azureml://registries/azureml/environments/sklearn-1.5/labels/latest",
    compute=config["compute_name"],
    experiment_name="train_student_success",
    display_name="student_success_prediction",
    limits=job_limits
)

# --- Submission ---
logger.info(f"Submitting workload to Azure Compute Cluster (Timeout: {TIMEOUT_SECONDS}s)...")
returned_job = ml_client.create_or_update(job)

os.makedirs("logs", exist_ok=True)
with open("logs/latest_job.txt", "w") as f:
    f.write(job_name)

logger.info(f"Submission Complete. Monitor progress via: {returned_job.studio_url}")
"""
Script Name: 01_submit_job.py
Author: Shane Lee
Description: Orchestrates the submission of the training workload to Azure Compute Clusters.
             Implements a build step to programmatically package local utility libraries
             into the source directory to ensure environment parity. Enforces execution
             timeouts via CommandJobLimits to prevent resource overconsumption.
"""

import json
import datetime
import csv
import os
import sys
import shutil
from azure.ai.ml import MLClient, command, Input
from azure.ai.ml.entities import CommandJobLimits
from azure.identity import DefaultAzureCredential

# --- Local Utils Import ---
import scalability_utils as utils

# --- Configuration & Logging ---

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
TEST_TRAIN_RATIO = hyperparams.get("test_train_ratio", 0.2)
LEARNING_RATE = hyperparams.get("learning_rate", 0.01)

# Serialise features to pass as argument
features_json_str = json.dumps(feature_defs)

credential = DefaultAzureCredential()

ml_client = MLClient(
    credential=credential,
    subscription_id=config["subscription_id"],
    resource_group_name=config["resource_group"],
    workspace_name=config["workspace_name"],
)

logger.info(f"Connection Established: {ml_client.workspace_name}")

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
base_name = config['job_base_name'].lower()
job_name = f"{base_name}-{timestamp}"

logger.info(f"Job Identifier Generated: {job_name}")

job_inputs = dict(
    data=Input(
        type="uri_file",
        path="https://raw.githubusercontent.com/srinivasav22/Graduate-Admission-Prediction/master/Admission_Predict_Ver1.1.csv",
    ),
    test_train_ratio=TEST_TRAIN_RATIO,
    learning_rate=LEARNING_RATE,
    features_json=features_json_str,
    registered_model_name="grad_admit_model",
)

job_limits = CommandJobLimits(timeout=TIMEOUT_SECONDS)

job = command(
    name=job_name,
    inputs=job_inputs,
    code="./src",
    command="python main.py --data ${{inputs.data}} --test_train_ratio ${{inputs.test_train_ratio}} --learning_rate ${{inputs.learning_rate}} --features_json '${{inputs.features_json}}' --registered_model_name ${{inputs.registered_model_name}}",
    environment="azureml://registries/azureml/environments/sklearn-1.5/labels/latest",
    compute=config["compute_name"],
    experiment_name="train_model_graduate_admissions",
    display_name="graduate_admissions_prediction",
    limits=job_limits
)

# --- Submission ---

logger.info(f"Submitting workload to Azure Compute Cluster (Timeout: {TIMEOUT_SECONDS}s)...")
returned_job = ml_client.create_or_update(job)

# --- Audit Logging ---

os.makedirs("logs", exist_ok=True)
ledger_file = "logs/job_history.csv"
file_exists = os.path.isfile(ledger_file)

ledger_data = [
    datetime.datetime.now().isoformat(),
    job_name,
    returned_job.status,
    job_inputs['learning_rate'],
    returned_job.studio_url
]

with open(ledger_file, "a", newline="") as f:
    writer = csv.writer(f)
    if not file_exists:
        writer.writerow(["Timestamp", "Job_ID", "Initial_Status", "Learning_Rate", "Job_URL"])
    writer.writerow(ledger_data)

with open("logs/latest_job.txt", "w") as f:
    f.write(job_name)

logger.info(f"Submission Complete. Monitor progress via: {returned_job.studio_url}")
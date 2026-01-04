"""
Script Name: 01_submit_job.py
Author: Shane Lee
Description: Orchestrates submission to Azure Compute.
             Packages shared utilities (Build Step) and passes NLP hyperparameters 
             to the cloud environment.
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
data_settings = config.get("data_settings", {})

TIMEOUT_SECONDS = ops_settings.get("training_timeout_seconds", 86400)
BATCH_SIZE = ops_settings.get("batch_size", 5000)

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
    logger.error(f"CRITICAL: Shared library {source_util} missing.")
    sys.exit(1)

# --- Job Definition ---
timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
job_name = f"{config['job_base_name']}-{timestamp}"

LOCAL_DATA_PATH = "inputs/feedback_stream_raw.csv"

if not os.path.exists(LOCAL_DATA_PATH):
    logger.error(f"CRITICAL: Training data {LOCAL_DATA_PATH} not found.")
    sys.exit(1)

job_inputs = dict(
    data=Input(type="uri_file", path=LOCAL_DATA_PATH),
    n_features=hyperparams.get("n_features", 1048576),
    epochs=hyperparams.get("epochs", 5),
    batch_size=BATCH_SIZE,
    text_col=data_settings.get("text_column", "Text"),
    label_col=data_settings.get("label_column", "Score"),
    threshold=data_settings.get("positive_threshold", 3),
    # UPDATED: Model Registry Name
    registered_model_name="feedback_classification_model",
)

job_limits = CommandJobLimits(timeout=TIMEOUT_SECONDS)

job = command(
    name=job_name,
    inputs=job_inputs,
    code="./src",
    command="python main.py --data ${{inputs.data}} --n_features ${{inputs.n_features}} --epochs ${{inputs.epochs}} --batch_size ${{inputs.batch_size}} --text_col ${{inputs.text_col}} --label_col ${{inputs.label_col}} --threshold ${{inputs.threshold}} --registered_model_name ${{inputs.registered_model_name}}",
    environment="azureml://registries/azureml/environments/sklearn-1.5/labels/latest",
    compute=config["compute_name"],
    # UPDATED: Experiment and Display Names
    experiment_name="train_feedback_classification",
    display_name="feedback_classification_nlp",
    limits=job_limits
)

# --- Submission ---
logger.info(f"Submitting workload to Azure Compute Cluster...")
returned_job = ml_client.create_or_update(job)

os.makedirs("logs", exist_ok=True)
with open("logs/latest_job.txt", "w") as f:
    f.write(job_name)

logger.info(f"Submission Complete. Monitor progress via: {returned_job.studio_url}")
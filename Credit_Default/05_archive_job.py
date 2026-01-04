"""
Script Name: 05_archive_job.py
Author: Shane Lee
Description: Lifecycle Management. Triggers a soft-delete operation on the Azure Machine Learning job.
             Implements error handling for platform-specific registry constraints.
"""

import json
import sys
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

# --- Local Utils Import ---
import scalability_utils as utils

# --- Setup & Logging ---

try:
    with open("config.json", "r") as f:
        config = json.load(f)
    utils.validate_config_version(config, expected_version="1.0")
except FileNotFoundError:
    print("CRITICAL: Configuration file missing.")
    sys.exit(1)
except Exception as e:
    print(f"CRITICAL: Configuration error: {e}")
    sys.exit(1)

logger = utils.configure_logging("ArchiveOps", "archived_jobs.log", config)

credential = DefaultAzureCredential()
ml_client = MLClient(
    credential=credential,
    subscription_id=config["subscription_id"],
    resource_group_name=config["resource_group"],
    workspace_name=config["workspace_name"],
)

# --- Execution ---

try:
    with open("logs/latest_job.txt", "r") as f:
        job_name = f.read().strip()
except FileNotFoundError:
    logger.error("Error: State file 'logs/latest_job.txt' not found. No active job to archive.")
    sys.exit(1)

logger.info(f"Attempting to archive job '{job_name}'...")

# --- Archival Logic with Platform Error Handling ---

try:
    # Trigger soft-delete in Azure.
    ml_client.jobs.archive(name=job_name)
    logger.info(f"SUCCESS: Job '{job_name}' archived.")

except Exception as e:
    error_msg = str(e)
    # Check for the specific Azure Registry bug (ArmScopeStructureInvalid)
    if "resource scope" in error_msg and "is not valid" in error_msg:
        logger.warning("WARNING: Azure refused to archive this job due to a Platform Limitation.")
        logger.warning("Reason: Jobs using Global Registry Environments cannot currently be archived via SDK.")
    else:
        logger.error(f"CRITICAL: Archive failed. Azure Error: {e}")
        sys.exit(1)
"""
Script Name: 05_archive_job.py
Author: Shane Lee
Description: Lifecycle Management. Archives the Azure Job.
             Includes specific exception handling for Azure Registry Environment limitations.
"""
import json
import sys
from azure.ai.ml import MLClient
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

logger = utils.configure_logging("ArchiveOps", "archived_jobs.log", config)

credential = DefaultAzureCredential()
ml_client = MLClient(
    credential=credential,
    subscription_id=config["subscription_id"],
    resource_group_name=config["resource_group"],
    workspace_name=config["workspace_name"],
)

try:
    with open("logs/latest_job.txt", "r") as f:
        job_name = f.read().strip()
    
    logger.info(f"Attempting to archive Job: {job_name}")
    ml_client.jobs.archive(name=job_name)
    logger.info(f"SUCCESS: Job archived.")

except Exception as e:
    error_str = str(e)
    # Specific handling for the Registry Environment Archive Bug
    # This suppresses the massive JSON error dump when using azureml://registries
    if "resource scope" in error_str and "registries" in error_str:
        logger.warning("WARNING: Platform Limitation - Jobs using Azure Registry Environments cannot be archived via SDK.")
        logger.warning(f"Job '{job_name}' remains visible in Studio but is considered complete by this pipeline.")
    else:
        # If it's any other error (e.g., Auth failed), we want to know about it.
        logger.error(f"CRITICAL: Archive failed. Error: {e}")
        sys.exit(1)
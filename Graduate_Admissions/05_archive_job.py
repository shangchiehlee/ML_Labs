"""
Script Name: 05_archive_job.py
Author: Shane Lee
Description: Lifecycle Management. Triggers a soft-delete operation on the Azure Job
             entity. Implements specific exception handling for Azure Registry
             environment limitations to distinguish between critical failures and
             known platform constraints.
"""

import json
import sys
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
import scalability_utils as utils

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
except FileNotFoundError:
    logger.error("Error: State file 'logs/latest_job.txt' not found.")
    sys.exit(1)

logger.info(f"Attempting to archive job '{job_name}'...")

try:
    ml_client.jobs.archive(name=job_name)
    logger.info(f"SUCCESS: Job '{job_name}' archived.")
except Exception as e:
    if "resource scope" in str(e):
        logger.warning("WARNING: Platform Limitation - Cannot archive Registry Environment jobs.")
    else:
        logger.error(f"CRITICAL: Archive failed. Error: {e}")
        sys.exit(1)
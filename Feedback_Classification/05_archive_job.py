"""Lifecycle management for cloud workloads.

Purpose:
    Archives completed Azure Machine Learning jobs using the Azure ML SDK.
    Handles registry-related SDK errors to avoid failing the local workflow.

Workflow:
    1. Identifies the target job ID from local state.
    2. Connects to Azure ML via `scalability_utils`.
    3. Attempts to archive the job.
    4. Logs a warning when the SDK reports a resource scope error.

Author: Shane Lee
Licence: MIT
"""

import logging
import sys
from typing import Any, Dict, Optional

# --- Local Utils Import ---
import scalability_utils as utils

# --- Configuration & Logging ---

config: Dict[str, Any] = utils.load_config()
logger: logging.Logger = utils.configure_logging("ArchiveOps", "archived_jobs.log", config)


def archive_job() -> None:
    """Attempts to archive the most recent cloud job via the Azure ML SDK.

    Logs a warning when the SDK reports a resource scope error. Exits with a
    fatal error for other archival failures.
    """
    job_name: Optional[str] = utils.get_latest_job_id()
    if not job_name:
        logger.error("No active job identified for archival in local state.")
        sys.exit(1)

    logger.info(f"Initiating archival for Job: {job_name}...")

    # Azure Connection Factory
    try:
        ml_client: Any = utils.get_azure_client(config)
    except Exception as e:
        logger.error(f"CRITICAL: Failed to connect to Azure ML Workspace: {e}")
        sys.exit(1)

    try:
        ml_client.jobs.archive(name=job_name)
        logger.info("SUCCESS: Job status updated to 'Archived'.")

    except Exception as e:
        error_str: str = str(e)
        # Handle registry-related SDK limitation signalled by a resource scope error.
        if "resource scope" in error_str.lower():
            logger.warning(
                "PLATFORM LIMITATION: Archive failed due to resource scope restrictions "
                "reported by the SDK."
            )
            logger.warning(f"Job '{job_name}' could not be archived via SDK.")
        else:
            logger.error(f"CRITICAL: Archival dispatch failed. Platform Error: {e}")
            sys.exit(1)


if __name__ == "__main__":
    archive_job()

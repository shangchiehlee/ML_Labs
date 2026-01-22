"""Lifecycle management for cloud workloads.

Purpose:
    Archives the most recent Azure Machine Learning job identified by
    utils.get_latest_job_id and handles registry-related exceptions returned
    by the SDK.

Workflow:
    1. Identifies the target job ID using utils.get_latest_job_id.
    2. Connects to Azure ML via `scalability_utils`.
    3. Requests archival via the Azure ML SDK.
    4. Handles registry limitations where archival is restricted.

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
    """Requests archival of the most recent cloud job returned by utils.get_latest_job_id.

    Requests archival of the job via the Azure ML SDK.
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
        # Handle registry-related SDK limitation signalled by a resource scope error
        if "resource scope" in error_str.lower():
            logger.warning(
                "PLATFORM LIMITATION: Jobs utilising Registry Environments "
                "cannot be archived via SDK."
            )
            logger.warning(
                f"Job '{job_name}' remains visible in Studio but is considered complete."
            )
        else:
            logger.error(f"CRITICAL: Archival dispatch failed. Platform Error: {e}")
            sys.exit(1)


if __name__ == "__main__":
    archive_job()

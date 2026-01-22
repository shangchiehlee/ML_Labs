"""Connectivity diagnostic utility for Azure Machine Learning.

Purpose:
    Attempts to resolve DefaultAzureCredential and performs a latency-measured
    API call to the target Workspace. Reports configuration values and
    connection status.

Workflow:
    1. Loads local configuration.
    2. Validates essential keys (subscription, resource group, workspace).
    3. Resolves Azure credentials using DefaultAzureCredential.
    4. Executes a lightweight API call (`workspaces.get`) to verify access.
    5. Reports latency and region data.

Author: Shane Lee
Licence: MIT
"""

import logging
import sys
import time
from typing import Any, Dict, Optional

# --- Local Utils Import ---
import scalability_utils as utils


def execute_handshake(ml_client: Any, workspace_name: str) -> str:
    """Performs a lightweight GET request to verify workspace accessibility.

    Args:
        ml_client: Authenticated MLClient instance.
        workspace_name: Name of the Azure ML Workspace.

    Returns:
        str: The geographic location of the workspace (e.g., "eastus").
    """
    workspace_obj: Any = ml_client.workspaces.get(workspace_name)
    return str(getattr(workspace_obj, "location", "Unknown Region"))


def main() -> None:
    """Main execution flow for connectivity diagnostics."""
    # --- Initialization ---
    config: Dict[str, Any] = utils.load_config()
    logger: logging.Logger = utils.configure_logging(
        "AzureConnectionForensics", "connection_ops.log", config
    )

    # --- Configuration Audit ---
    sub_id: Optional[str] = config.get("subscription_id")
    rg_name: Optional[str] = config.get("resource_group")
    ws_name: Optional[str] = config.get("workspace_name")

    logger.info("--- CONFIGURATION INTEGRITY CHECK ---")
    logger.info(f"Subscription ID : {sub_id}")
    logger.info(f"Resource Group  : {rg_name}")
    logger.info(f"Workspace Name  : {ws_name}")

    if not all([sub_id, rg_name, ws_name]):
        logger.error("CRITICAL: Missing essential configuration parameters.")
        sys.exit(1)

    # --- Resource Inspection ---
    utils.get_memory_limit()

    # --- Authentication & Connection ---
    logger.info("--- INITIATING HANDSHAKE ---")
    start_time: float = time.time()
    connection_status: str = "UNKNOWN"
    details: str = "N/A"

    try:
        # Identity Negotiation
        logger.info("Resolving DefaultAzureCredential chain...")
        ml_client: Any = utils.get_azure_client(config)
        
        logger.info("Credential resolved. Attempting API handshake...")

        # Verification Call
        location: str = execute_handshake(ml_client, str(ws_name))
        connection_status = "SUCCESS"
        details = f"Region: {location}"

    except Exception as e:
        connection_status = "FAILED"
        details = str(e)
        logger.error(f"CONNECTION FAILURE: {details}")
        logger.info("REMEDIAL ACTION: Verify 'az login' and network egress rules.")

    finally:
        duration: float = round(time.time() - start_time, 2)

    # --- Scorecard Output ---
    logger.info(" ")
    logger.info("========================================")
    logger.info("   CONNECTIVITY HEALTH SCORECARD        ")
    logger.info("========================================")
    logger.info(f"Target Workspace  : {ws_name}")
    logger.info(f"Execution Latency : {duration}s")
    logger.info(f"Final Status      : {connection_status}")
    logger.info(f"Artifacts/Details : {details}")
    logger.info("========================================")

    if connection_status != "SUCCESS":
        sys.exit(1)


if __name__ == "__main__":
    main()

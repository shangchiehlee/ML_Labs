"""
Script Name: connect_to_workspace.py
Author: Shane Lee
Description: Validates the network and authentication path to the Azure Machine 
             Learning Control Plane. Utilises DefaultAzureCredential to verify 
             Identity Access Management (IAM) policies and measures connection 
             latency to the target Workspace.
Key Outputs:
    - logs/connection_ops.log
"""

import sys
import time
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
import scalability_utils as utils

def main():
    # --- Initialization ---
    config = utils.load_configuration()
    logger = utils.setup_logger("AzureConnectionForensics", "connection_ops.log", config)

    # --- Configuration Audit ---
    sub_id = config.get("subscription_id")
    rg_name = config.get("resource_group")
    ws_name = config.get("workspace_name")

    logger.info("--- CONFIGURATION INTEGRITY CHECK ---")
    logger.info(f"Targeting Subscription ID : {sub_id}")
    logger.info(f"Targeting Resource Group  : {rg_name}")
    logger.info(f"Targeting Workspace Name  : {ws_name}")

    if not sub_id or not rg_name or not ws_name:
        logger.error("CRITICAL: Missing configuration parameters.")
        sys.exit(1)

    # --- Resource Inspection (Container Awareness) ---
    utils.check_container_memory(logger)

    # --- Authentication & Connection ---
    logger.info("--- INITIATING HANDSHAKE ---")
    start_time = time.time()
    connection_status = "UNKNOWN"
    details = ""

    try:
        # Identity Negotiation
        logger.info("Attempting to resolve DefaultAzureCredential...")
        credential = DefaultAzureCredential()
        logger.info("   -> Action: Credential object initialised.")

        # Client Instantiation
        logger.info("Constructing MLClient interface...")
        ml_client = MLClient(
            credential=credential,
            subscription_id=sub_id,
            resource_group_name=rg_name,
            workspace_name=ws_name,
        )

        # Connection Verification (Latency Check)
        workspace_obj = ml_client.workspaces.get(ws_name)
        
        connection_status = "SUCCESS"
        details = f"Location: {workspace_obj.location}"

    except Exception as e:
        connection_status = "FAILED"
        details = str(e)
        logger.error("CONNECTION FAILURE DETECTED")
        logger.error(f"   -> Evidence: {details}")
        logger.error("   -> Action: Check 'az login' status or firewall rules.")

    finally:
        end_time = time.time()
        duration = round(end_time - start_time, 2)

    # --- Connectivity Health Scorecard ---
    logger.info(" ")
    logger.info("========================================")
    logger.info("   CONNECTIVITY HEALTH SCORECARD        ")
    logger.info("========================================")
    logger.info(f"Target Workspace  : {ws_name}")
    logger.info(f"Execution Time    : {duration} seconds")
    logger.info(f"Final Status      : {connection_status}")

    if connection_status == "SUCCESS":
        logger.info(f"Details           : {details}")
        logger.info("Result            : READY FOR OPERATIONS")
    else:
        logger.error("Result            : OPERATIONAL STOPPAGE")

    logger.info("========================================")

if __name__ == "__main__":
    main()
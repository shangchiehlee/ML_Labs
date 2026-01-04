"""
Script Name: 02_download_outputs.py
Author: Shane Lee
Description: Retrieves serialised artifacts and operational logs from Azure Blob Storage.
             Executes a zero-byte integrity check on all downloaded files to detect
             transfer corruption. Generates a JSON manifest to establish the chain of
             custody for the retrieved assets.
"""

import os
import json
import datetime
import sys
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

# --- Local Utils Import ---
import scalability_utils as utils

# --- Setup ---

try:
    with open("config.json", "r") as f:
        config = json.load(f)
    utils.validate_config_version(config, expected_version="1.0")
except Exception as e:
    print(f"CRITICAL: Configuration error: {e}")
    sys.exit(1)

logger = utils.configure_logging("DownloadForensics", "download_ops.log", config)

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
    logger.info(f"Target Job Identified: {job_name}")
except FileNotFoundError:
    logger.error("CRITICAL: State file 'logs/latest_job.txt' missing.")
    sys.exit(1)

# --- Retrieval ---

local_path = f"./downloaded_artifacts/{job_name}"
logger.info(f"Initialising download to: {local_path}...")

try:
    ml_client.jobs.download(name=job_name, download_path=local_path, all=True)
    logger.info("API transfer completed successfully.")
except Exception as e:
    logger.error(f"CRITICAL: Transfer failure. Azure Error: {e}")
    sys.exit(1)

# --- Integrity Check ---

logger.info("--- COMMENCING INTEGRITY CHECK ---")

manifest = {
    "job_id": job_name,
    "download_timestamp": datetime.datetime.now().isoformat(),
    "files": [],
    "integrity_warnings": []
}

required_files = ["forensic_audit.log", "model.joblib"]
found_files = set()

for root, dirs, files in os.walk(local_path):
    for filename in files:
        full_path = os.path.join(root, filename)
        file_size = os.path.getsize(full_path)
        
        file_entry = {"path": full_path, "size_bytes": file_size, "status": "VALID"}
        
        if file_size == 0:
            logger.warning(f"INTEGRITY WARNING: Zero-byte artifact detected: {filename}")
            file_entry["status"] = "CORRUPT (0 Bytes)"
            manifest["integrity_warnings"].append(filename)
        
        if filename in required_files:
            found_files.add(filename)
            logger.info(f"Verified Critical Artifact: {filename} ({file_size} bytes)")

        manifest["files"].append(file_entry)

missing_critical = set(required_files) - found_files

if missing_critical:
    logger.error(f"CRITICAL: Mandatory artifacts missing: {missing_critical}")
    manifest["status"] = "INCOMPLETE"
else:
    logger.info("SUCCESS: All critical artifacts verified.")
    manifest["status"] = "COMPLETE"

manifest_path = os.path.join(local_path, "download_manifest.json")
with open(manifest_path, "w") as f:
    json.dump(manifest, f, indent=4)

logger.info(f"Chain of Custody Manifest saved to: {manifest_path}")
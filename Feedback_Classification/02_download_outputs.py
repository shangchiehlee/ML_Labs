"""
Script Name: 02_download_outputs.py
Author: Shane Lee
Description: Retrieves serialised artifacts and operational logs.
             Executes zero-byte integrity check.
             Implements 'Publishing Pattern' to promote Cloud-Generated 
             Business Intelligence CSVs to the local 'outputs/' directory.
"""

import os
import json
import datetime
import sys
import shutil
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

# Path where Azure downloads raw artifacts (Audit Trail)
artifact_path = f"./downloaded_artifacts/{job_name}"
logger.info(f"Initialising download to: {artifact_path}...")

try:
    ml_client.jobs.download(name=job_name, download_path=artifact_path, all=True)
    logger.info("API transfer completed successfully.")
except Exception as e:
    logger.error(f"CRITICAL: Transfer failure. Azure Error: {e}")
    sys.exit(1)

# --- Integrity Check ---
manifest = {
    "job_id": job_name,
    "download_timestamp": datetime.datetime.now().isoformat(),
    "files": [],
    "status": "PENDING"
}

required_files = ["forensic_audit.log", "model.joblib"]
found_files = set()

for root, dirs, files in os.walk(artifact_path):
    for filename in files:
        full_path = os.path.join(root, filename)
        file_size = os.path.getsize(full_path)
        file_entry = {"path": full_path, "size_bytes": file_size, "status": "VALID"}
        
        if file_size == 0:
            logger.warning(f"INTEGRITY WARNING: Zero-byte artifact: {filename}")
            file_entry["status"] = "CORRUPT"
        
        if filename in required_files:
            found_files.add(filename)
            logger.info(f"Verified Critical Artifact: {filename} ({file_size} bytes)")

        manifest["files"].append(file_entry)

if set(required_files) - found_files:
    logger.error(f"CRITICAL: Mandatory artifacts missing.")
    manifest["status"] = "INCOMPLETE"
else:
    logger.info("SUCCESS: All critical artifacts verified.")
    manifest["status"] = "COMPLETE"

with open(os.path.join(artifact_path, "download_manifest.json"), "w") as f:
    json.dump(manifest, f, indent=4)

# --- PUBLISHING STEP (Fixed) ---
# We promote the Business Intelligence CSVs from the artifact storage 
# to the local 'outputs/' folder.

LOCAL_OUTPUT_DIR = "outputs"
os.makedirs(LOCAL_OUTPUT_DIR, exist_ok=True)

# Azure SDK v2 nests downloads inside an 'artifacts' folder.
# We check both possible paths to be robust.
possible_paths = [
    os.path.join(artifact_path, "artifacts", "outputs"), # Standard v2 path
    os.path.join(artifact_path, "outputs")               # Fallback
]

source_output_dir = None
for path in possible_paths:
    if os.path.exists(path):
        source_output_dir = path
        break

reports_to_publish = [
    "priority_alerts.csv",
    "sentiment_drivers.csv",
    "forensic_audit_full.csv"
]

logger.info("--- PUBLISHING BUSINESS REPORTS ---")

if source_output_dir:
    logger.info(f"Source Outputs Detected: {source_output_dir}")
    for report in reports_to_publish:
        src = os.path.join(source_output_dir, report)
        dst = os.path.join(LOCAL_OUTPUT_DIR, report)
        
        if os.path.exists(src):
            shutil.copy2(src, dst)
            logger.info(f"Published to Workspace: {dst}")
        else:
            logger.warning(f"Report not found in artifacts: {report}")
else:
    logger.warning(f"Could not locate 'outputs' folder in {artifact_path}")

logger.info("Download and Publishing Complete.")
"""Retrieval and forensic audit utility for cloud artefacts.

Purpose:
    Retrieves serialised model artefacts and operational logs from Azure ML job
    outputs. Executes zero-byte integrity checks on all downloaded files and
    generates a local manifest of retrieved files.

Workflow:
    1. Identifies the latest job ID from local state.
    2. Downloads all artefacts to `downloaded_artifacts/<job_id>`.
    3. Scans for zero-byte files and verifies mandatory assets (`model.joblib`).
    4. Generates a `download_manifest.json` ledger.

Author: Shane Lee
Licence: MIT
"""

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, TypedDict

# --- Local Utils Import ---
import scalability_utils as utils

# --- Configuration & Logging ---

config: Dict[str, Any] = utils.load_config()
logger: logging.Logger = utils.configure_logging("DownloadForensics", "download_ops.log", config)


# --- Type Definitions ---

class FileEntry(TypedDict):
    """Metadata schema for a retrieved file artefact."""
    path: str
    size_bytes: int
    status: str


class Manifest(TypedDict):
    """Audit schema for the retrieval operation."""
    job_id: str
    files: List[FileEntry]
    integrity_warnings: List[str]


# --- Core Logic ---

def perform_integrity_check(local_path: Path, job_name: str) -> Manifest:
    """Scans retrieved artefacts for zero-byte files or missing required assets.

    Args:
        local_path: Directory containing the downloaded artefacts.
        job_name: ID of the cloud job being verified.

    Returns:
        Manifest: A dictionary containing the audit results and file metadata.
    """
    logger.info("--- COMMENCING ARTIFACT INTEGRITY CHECK ---")

    manifest: Manifest = {
        "job_id": job_name,
        "files": [],
        "integrity_warnings": [],
    }

    # Assets required for integrity checks
    required_files: Set[str] = {
        "forensic_audit.log",
        utils.ProjectPaths.MODEL_FILENAME
    }
    found_files: Set[str] = set()

    for file_path in sorted(local_path.rglob("*"), key=lambda p: str(p)):
        if not file_path.is_file():
            continue

        file_size: int = file_path.stat().st_size
        file_entry: FileEntry = {
            "path": str(file_path.relative_to(local_path)),
            "size_bytes": file_size,
            "status": "VALID"
        }

        # Zero-byte detection
        if file_size == 0:
            logger.warning(f"INTEGRITY WARNING: Zero-byte artifact detected: {file_path.name}")
            file_entry["status"] = "CORRUPT (0 Bytes)"
            manifest["integrity_warnings"].append(file_path.name)

        if file_path.name in required_files:
            found_files.add(file_path.name)
            logger.info(f"Verified Mandatory Asset: {file_path.name} [{file_size} bytes]")

        manifest["files"].append(file_entry)

    # Final Audit Verdict
    missing_critical: Set[str] = required_files - found_files

    if missing_critical:
        logger.error(f"CRITICAL: Mandatory artifacts missing from retrieval: {missing_critical}")
    else:
        logger.info("SUCCESS: All mandatory artifacts verified.")

    return manifest


def main() -> None:
    """Main execution flow for artefact retrieval and verification."""
    job_name: Optional[str] = utils.get_latest_job_id()
    if not job_name:
        logger.error("CRITICAL: No active job identifier found in local state.")
        sys.exit(1)

    logger.info(f"Synchronising artifacts for Job: {job_name}")

    # Azure Connection Factory
    try:
        ml_client: Any = utils.get_azure_client(config)
    except Exception as e:
        logger.error(f"CRITICAL: Failed to connect to Azure ML Workspace: {e}")
        sys.exit(1)

    # Artifact Retrieval
    download_dir: Path = utils.ProjectPaths.ARTIFACTS / job_name
    logger.info(f"Initialising API transfer to: {download_dir}...")

    try:
        ml_client.jobs.download(
            name=job_name, download_path=str(download_dir), all=True
        )
        logger.info("Transfer synchronisation completed successfully.")
    except Exception as e:
        logger.error(f"CRITICAL: Transfer failure. Azure Platform Error: {e}")
        sys.exit(1)

    # Forensic Audit
    manifest_data: Manifest = perform_integrity_check(download_dir, job_name)

    # Ledger Update
    manifest_path: Path = download_dir / "download_manifest.json"
    try:
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest_data, f, indent=4)
        logger.info(f"Forensic manifest generated: {manifest_path}")
    except Exception as e:
        logger.error(f"Failed to update manifest ledger: {e}")

    required_files: Set[str] = {
        "forensic_audit.log",
        utils.ProjectPaths.MODEL_FILENAME
    }
    found_files: Set[str] = {Path(entry["path"]).name for entry in manifest_data["files"]}
    missing_critical: Set[str] = required_files - found_files
    if missing_critical:
        logger.error("WORKFLOW TERMINATED: Artifact integrity check failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()

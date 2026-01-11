"""Retrieval and Forensic Audit Utility for Cloud Artifacts.

Purpose:
    Retrieves serialised model artifacts and operational logs from Azure Blob Storage.
    Executes zero-byte integrity checks on all downloaded files to detect transfer
    corruption and generates a local manifest to establish the chain of custody.

Workflow:
    1. Identifies the latest job ID from local state.
    2. Downloads all artifacts to `downloaded_artifacts/<job_id>`.
    3. Scans for zero-byte files and verifies mandatory assets (`model.joblib`).
    4. Generates a `download_manifest.json` ledger.
    5. Promotes business intelligence reports to the `outputs/` directory.

Author: Shane Lee
Licence: MIT
"""

import datetime
import json
import logging
import shutil
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
    """Metadata schema for a retrieved file artifact."""
    path: str
    size_bytes: int
    status: str


class Manifest(TypedDict):
    """Audit schema for the retrieval operation."""
    job_id: str
    download_timestamp: str
    files: List[FileEntry]
    integrity_warnings: List[str]
    status: str


# --- Core Logic ---

def perform_integrity_check(local_path: Path, job_name: str) -> Manifest:
    """Scans retrieved artifacts for corruption or mandatory asset omission.

    Args:
        local_path: Directory containing the downloaded artifacts.
        job_name: ID of the cloud job being verified.

    Returns:
        Manifest: A dictionary containing the audit results and file metadata.
    """
    logger.info("--- COMMENCING ARTIFACT INTEGRITY CHECK ---")

    manifest: Manifest = {
        "job_id": job_name,
        "download_timestamp": datetime.datetime.now().isoformat(),
        "files": [],
        "integrity_warnings": [],
        "status": "PENDING"
    }

    # Assets mandatory for pipeline promotion
    required_files: Set[str] = {
        "forensic_audit_full.csv",
        utils.ProjectPaths.MODEL_FILENAME
    }
    found_files: Set[str] = set()

    for file_path in local_path.rglob("*"):
        if not file_path.is_file():
            continue

        file_size: int = file_path.stat().st_size
        file_entry: FileEntry = {
            "path": str(file_path.relative_to(local_path)),
            "size_bytes": file_size,
            "status": "VALID"
        }

        # Transfer Corruption Detection
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
        manifest["status"] = "INCOMPLETE"
    else:
        logger.info("SUCCESS: All mandatory artifacts verified.")
        manifest["status"] = "COMPLETE"

    return manifest


def publish_reports(artifact_path: Path, output_dir: Path) -> None:
    """Promotes validated business intelligence reports to the production output directory.

    Args:
        artifact_path: Source directory of downloaded assets.
        output_dir: Target directory for local report promotion.
    """
    output_dir.mkdir(exist_ok=True)

    reports: List[str] = [
        "priority_alerts.csv",
        "sentiment_drivers.csv",
        "forensic_audit_full.csv"
    ]
    source_found: Optional[Path] = None

    # Resolve standard Azure SDK nesting patterns (artifacts/outputs vs outputs)
    search_dirs: List[Path] = [
        artifact_path / "artifacts" / "outputs",
        artifact_path / "outputs",
        artifact_path
    ]

    for candidate in search_dirs:
        if candidate.exists() and (candidate / "forensic_audit_full.csv").exists():
            source_found = candidate
            break

    if not source_found:
        logger.warning("Target 'outputs' subdirectory not detected in artifacts. Manual extraction required.")
        return

    logger.info(f"Promoting reports from: {source_found}")
    for r in reports:
        src: Path = source_found / r
        dst: Path = output_dir / r
        if src.exists():
            shutil.copy2(src, dst)
            logger.info(f"Published: {dst.name}")


def main() -> None:
    """Main execution flow for artifact retrieval and verification."""
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

    # Report Promotion
    publish_reports(download_dir, utils.ProjectPaths.OUTPUTS)

    if manifest_data["status"] != "COMPLETE":
        logger.error("WORKFLOW TERMINATED: Artifact integrity check failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
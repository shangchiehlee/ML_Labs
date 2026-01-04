"""
Script Name: 06_cleanup_local.py
Author: Shane Lee
Description: Workspace Sanitisation. Recursively removes ephemeral data directories, logs, and build artifacts.
             Utilises console-only logging to prevent file locking issues during directory deletion.
"""

import os
import shutil
import sys
import logging

# --- Console-Only Logging Setup ---
# We do NOT use utils.configure_logging here because it creates a file lock in 'logs/'
logger = logging.getLogger("CleanupOps")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)

logger.info("--- STARTING WORKSPACE SANITISATION (HARD RESET) ---")

# --- 1. Directory Removal (Recursive) ---
# These directories contain data, logs, and artifacts. They will be fully removed.
directories_to_clean = [
    "inputs",
    "outputs",
    "logs",
    "downloaded_artifacts"
]

for target_dir in directories_to_clean:
    if os.path.exists(target_dir):
        try:
            shutil.rmtree(target_dir)
            logger.info(f"Deleted directory recursively: {target_dir}")
        except Exception as e:
            logger.error(f"Error deleting {target_dir}: {e}")
    else:
        logger.info(f"Skipped: {target_dir} (Not found)")

# --- 2. Temporary Build Artifacts Removal ---
# We remove the copy of scalability_utils.py from src/ to prevent
# accidental editing of the wrong file.
temp_util_copy = "src/scalability_utils.py"

if os.path.exists(temp_util_copy):
    try:
        os.remove(temp_util_copy)
        logger.info(f"Deleted temporary build artifact: {temp_util_copy}")
    except Exception as e:
        logger.error(f"Error deleting {temp_util_copy}: {e}")
else:
    logger.info(f"Skipped: {temp_util_copy} (Not found)")

# --- 3. Python Bytecode & Cache Removal ---
# Recursively find and remove __pycache__ directories
logger.info("Scanning for Python bytecode caches...")

cache_count = 0
for root, dirs, files in os.walk("."):
    for d in dirs:
        if d == "__pycache__":
            cache_path = os.path.join(root, d)
            try:
                shutil.rmtree(cache_path)
                logger.info(f"Deleted cache: {cache_path}")
                cache_count += 1
            except Exception as e:
                logger.warning(f"Failed to delete {cache_path}: {e}")

if cache_count == 0:
    logger.info("No bytecode caches found.")

logger.info("--- SANITISATION COMPLETE ---")
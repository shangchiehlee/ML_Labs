"""
Script Name: 06_cleanup_local.py
Author: Shane Lee
Description: Executes a sanitisation routine to remove transient runtime artifacts. 
             Deletes the 'logs/' directory and recursively removes '__pycache__' 
             bytecode folders. Utilises console-only logging to prevent OS-level 
             file locking errors during directory deletion.
"""

import sys
import os
import shutil
import logging

# --- Console-Only Logging Configuration ---
# CRITICAL: Do not add a FileHandler here. 
# Writing to a file inside the directory we are about to delete causes 
# OS-level locking errors (PermissionError), especially on Windows.

logger = logging.getLogger("LocalCleanup")
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

def main():
    logger.info("--- LOCAL ENVIRONMENT CLEANUP ---")
    
    # 1. Delete the explicit 'logs' directory
    log_dir = "logs"
    if os.path.exists(log_dir):
        try:
            shutil.rmtree(log_dir)
            logger.info(f"Deleted Directory: {log_dir}/")
        except Exception as e:
            logger.error(f"Failed to delete '{log_dir}': {e}")
    else:
        logger.info(f"Target not found (already clean): {log_dir}/")

    # 2. Recursively find and delete all '__pycache__' directories
    # These are generated automatically by Python in every folder containing imported scripts.
    logger.info("Scanning for bytecode caches (__pycache__)...")
    
    cache_count = 0
    for root, dirs, files in os.walk("."):
        for d in dirs:
            if d == "__pycache__":
                path = os.path.join(root, d)
                try:
                    shutil.rmtree(path)
                    logger.info(f"Deleted Cache: {path}")
                    cache_count += 1
                except Exception as e:
                    logger.error(f"Failed to delete '{path}': {e}")

    if cache_count == 0:
        logger.info("No bytecode caches found.")

    logger.info("-" * 40)
    logger.info("Cleanup Complete. Environment ready for fresh execution.")

if __name__ == "__main__":
    main()
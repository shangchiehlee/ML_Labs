"""
Script Name: 06_cleanup_local.py
Author: Shane Lee
Description: Workspace Sanitisation. Hard reset of local environment.
             CRITICAL: Uses Console-Only Logging to avoid OS file locks on the 'logs/' directory.
"""
import os
import shutil
import stat
import time

def console_log(message, level="INFO"):
    """
    Standalone logger that prints to console only.
    Prevents file locking issues in the logs directory.
    """
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"{timestamp} - {level} - {message}")

def remove_readonly(func, path, excinfo):
    """
    Error handler for shutil.rmtree. Forces write permissions.
    """
    try:
        os.chmod(path, stat.S_IWRITE)
        func(path)
    except Exception as e:
        console_log(f"Failed to force delete {path}: {e}", "ERROR")

def force_delete_directory(target_dir):
    if os.path.exists(target_dir):
        console_log(f"Targeting directory: {target_dir}...")
        try:
            shutil.rmtree(target_dir, onerror=remove_readonly)
            console_log(f"DELETED: {target_dir}")
        except Exception as e:
            console_log(f"FAILED to delete {target_dir}. Reason: {e}", "ERROR")

def force_delete_file(target_file):
    if os.path.exists(target_file):
        try:
            os.chmod(target_file, stat.S_IWRITE)
            os.remove(target_file)
            console_log(f"DELETED: {target_file}")
        except Exception as e:
            console_log(f"FAILED to delete {target_file}. Reason: {e}", "ERROR")

if __name__ == "__main__":
    console_log("--- STARTING WORKSPACE SANITISATION ---")

    # 1. Artifacts & Logs (High Priority)
    force_delete_directory("downloaded_artifacts")
    force_delete_directory("logs")

    # 2. Data Directories
    force_delete_directory("inputs")
    force_delete_directory("outputs")

    # 3. Build Artifacts
    force_delete_file("src/scalability_utils.py")
    
    # 4. Bytecode
    for root, dirs, files in os.walk("."):
        for d in dirs:
            if d == "__pycache__":
                shutil.rmtree(os.path.join(root, d), onerror=remove_readonly)

    console_log("--- SANITISATION COMPLETE ---")
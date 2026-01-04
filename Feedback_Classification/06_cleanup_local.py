"""
Script Name: 06_cleanup_local.py
Author: Shane Lee
Description: Workspace Sanitisation. Hard reset of local environment.
             Uses Console-Only Logging to avoid OS file locks on the 'logs/' directory.
"""
import os
import shutil
import stat
import time

def console_log(message, level="INFO"):
    """Standalone logger that prints to console only."""
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {level} - {message}")

def remove_readonly(func, path, excinfo):
    """Error handler for shutil.rmtree. Forces write permissions."""
    try:
        os.chmod(path, stat.S_IWRITE)
        func(path)
    except Exception:
        pass

def force_delete_directory(target_dir):
    if os.path.exists(target_dir):
        console_log(f"Deleting: {target_dir}...")
        shutil.rmtree(target_dir, onerror=remove_readonly)

def force_delete_file(target_file):
    if os.path.exists(target_file):
        try:
            os.remove(target_file)
            console_log(f"Deleted: {target_file}")
        except Exception:
            pass

if __name__ == "__main__":
    console_log("--- STARTING WORKSPACE SANITISATION ---")
    
    # 1. Directories
    force_delete_directory("downloaded_artifacts")
    force_delete_directory("logs")
    force_delete_directory("inputs")
    force_delete_directory("outputs")
    
    # 2. Build Artifacts
    force_delete_file("src/scalability_utils.py")
    
    # 3. Bytecode
    for root, dirs, files in os.walk("."):
        for d in dirs:
            if d == "__pycache__":
                shutil.rmtree(os.path.join(root, d), onerror=remove_readonly)

    console_log("--- SANITISATION COMPLETE ---")
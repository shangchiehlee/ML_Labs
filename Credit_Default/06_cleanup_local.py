"""Workspace sanitisation utility.

Purpose:
    Removes transient data directories, logs, and build artefacts from the
    local environment.

Safety Mechanisms:
    *   Uses standard library utilities only.
    *   Implements recursive permission modification to handle read-only artefacts.
    *   Avoids file handlers for logging to reduce log file locking risk.

Author: Shane Lee
Licence: MIT
"""

import os
import shutil
import stat
import time
from pathlib import Path
from typing import Any, Callable


def console_log(message: str, level: str = "INFO") -> None:
    """Standalone logger that prints to console only.

    Prevents file locking issues in the logs directory by avoiding the use of
    FileHandlers during the sanitisation phase.

    Args:
        message: Log message.
        level: Log level (INFO, ERROR, etc.).
    """
    timestamp: str = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"{timestamp} - {level} - {message}")


def remove_readonly(func: Callable, path: str, excinfo: Any) -> None:
    """Error handler for shutil.rmtree to manage restricted permissions.

    If a file is read-only, this forces write permissions and retries the deletion.

    Args:
        func: The function that failed (e.g., os.unlink).
        path: The path to the file that caused the error.
        excinfo: Exception information.
    """
    try:
        os.chmod(path, stat.S_IWRITE)
        func(path)
    except Exception as e:
        console_log(f"Failed to force delete {path}: {e}", "ERROR")


def force_delete_directory(target_dir: Path) -> None:
    """Recursively deletes a directory with retry logic and permission handling.

    Args:
        target_dir: Path object to directory to delete.
    """
    if target_dir.exists():
        console_log(f"Targeting directory: {target_dir}...")
        try:
            shutil.rmtree(target_dir, onerror=remove_readonly)
            console_log(f"DELETED: {target_dir}")
        except Exception as e:
            console_log(f"FAILED to delete {target_dir}. Reason: {e}", "ERROR")
    else:
        console_log(f"Skipped: {target_dir} (Not found)")


def force_delete_file(target_file: Path) -> None:
    """Deletes a single file with forced write permissions.

    Args:
        target_file: Path object to file to delete.
    """
    if target_file.exists():
        try:
            # Ensure write permissions before unlinking
            os.chmod(target_file, stat.S_IWRITE)
            target_file.unlink()
            console_log(f"DELETED: {target_file}")
        except Exception as e:
            console_log(f"FAILED to delete {target_file}. Reason: {e}", "ERROR")
    else:
        console_log(f"Skipped: {target_file} (Not found)")


def clean_pycache() -> None:
    """Recursively identifies and purges Python bytecode caches."""
    console_log("Scanning for Python bytecode caches...")
    count: int = 0
    root: Path = Path(".")

    for path in root.rglob("__pycache__"):
        if path.is_dir():
            shutil.rmtree(path, onerror=remove_readonly)
            count += 1

    console_log(f"Cleaned {count} bytecode cache directories.")


def main() -> None:
    """Main sanitisation execution flow."""
    console_log("--- STARTING WORKSPACE SANITISATION ---")

    # 1. Ephemeral Data & Logs
    force_delete_directory(Path("downloaded_artifacts"))
    force_delete_directory(Path("logs"))
    force_delete_directory(Path("inputs"))
    force_delete_directory(Path("outputs"))

    # 2. Build Artifacts
    # Removes the injected utility from the source dir
    force_delete_file(Path("src") / "scalability_utils.py")

    # 3. Bytecode Cache
    clean_pycache()

    console_log("--- SANITISATION COMPLETE ---")


if __name__ == "__main__":
    main()

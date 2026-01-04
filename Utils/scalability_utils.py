"""
Script Name: scalability_utils.py
Author: Shane Lee
Description: A shared library containing infrastructure primitives. Implements 
             Cgroup-aware memory detection for containerised environments, a 
             fault-tolerant JSON configuration loader, and a standardised logging 
             factory. Serves as a strict dependency for environment verification scripts.
"""

import sys
import os
import json
import logging
import psutil

def load_configuration(config_path="config.json"):
    """
    Loads and validates the external JSON configuration.
    """
    if not os.path.exists(config_path):
        print(f"CRITICAL: Configuration file '{config_path}' missing.")
        sys.exit(1)
        
    with open(config_path, "r") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError as e:
            print(f"CRITICAL: Invalid JSON in '{config_path}': {e}")
            sys.exit(1)

def setup_logger(logger_name, log_filename, config):
    """
    Configures a dual-output logger (Console + File).
    Enforces artifact isolation by writing to a 'logs/' directory.
    """
    # Retrieve settings from config or use defaults
    ops_config = config.get("operations", {})
    log_dir = ops_config.get("log_directory", "logs")
    
    # Ensure log directory exists (Artifact Isolation)
    os.makedirs(log_dir, exist_ok=True)
    full_log_path = os.path.join(log_dir, log_filename)

    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    
    # Clear existing handlers to prevent duplicate logs during re-runs
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File Handler
    file_handler = logging.FileHandler(full_log_path, mode='w')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger

def check_container_memory(logger):
    """
    Detects memory limits, prioritising Cgroups (Container) over Host OS.
    Crucial for Azure ML Compute stability.
    """
    # 1. Check Cgroup Limit (Docker/Kubernetes/AzureML)
    cgroup_mem_path = "/sys/fs/cgroup/memory/memory.limit_in_bytes"
    
    if os.path.isfile(cgroup_mem_path):
        try:
            with open(cgroup_mem_path, "r") as f:
                mem_bytes = int(f.read().strip())
                # Cgroups often report a very large number for "unlimited"
                if mem_bytes < 1e15: 
                    mem_gb = mem_bytes / (1024**3)
                    logger.info(f"Environment: Container Detected. Memory Limit: {mem_gb:.2f} GB")
                    return
        except Exception:
            pass # Fallback to psutil if cgroup read fails

    # 2. Fallback to Host Memory (Local Machine)
    mem_gb = psutil.virtual_memory().total / (1024**3)
    logger.info(f"Environment: Host OS Detected. Total Memory: {mem_gb:.2f} GB")
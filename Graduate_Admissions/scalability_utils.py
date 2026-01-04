"""
Script Name: scalability_utils.py
Author: Shane Lee
Description: Shared Infrastructure Library. Provides container-aware resource detection
             by inspecting Linux Cgroups (v1/v2) to identify true memory limits.
             Implements structured JSON logging for enterprise ingestion and
             probabilistic data structures (Bloom Filters) for O(1) deduplication.
"""

import shutil
import psutil
import logging
import json
import sys
import os
import datetime
import math
import hashlib
import pandas as pd

# --- Logging Utilities ---

class JSONFormatter(logging.Formatter):
    """
    Formatter to output logs in JSON format for enterprise monitoring tools (Splunk, Datadog).
    """
    def format(self, record):
        log_record = {
            "timestamp": datetime.datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "line": record.lineno
        }
        if record.exc_info:
            log_record["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_record)

def configure_logging(name, log_filename, config=None):
    """
    Configures a standardised logger with dual output (Console + File).
    Automatically places log files in a 'logs/' directory.
    """
    # Ensure logs directory exists
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_filename)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers = []  # Clear existing handlers to prevent duplication

    # Determine format based on config
    use_json = False
    if config and config.get("operational_settings", {}).get("enable_json_logging"):
        use_json = True

    if use_json:
        formatter = JSONFormatter()
    else:
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File Handler
    file_handler = logging.FileHandler(log_path, mode='a')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger

# --- Configuration Validation ---

def validate_config_version(config, expected_version="1.0"):
    """
    Validates that the configuration file matches the expected schema version.
    """
    version = config.get("schema_version", "0.0")
    if version != expected_version:
        error_msg = f"Config Version Mismatch: Expected {expected_version}, found {version}. Please update config.json."
        print(f"CRITICAL: {error_msg}", file=sys.stderr)
        raise ValueError(error_msg)
    return True

# --- Resource Management Utilities ---

# Configure module-level logger (NullHandler to avoid noise if not configured)
logger = logging.getLogger("ScalabilityUtils")
logger.addHandler(logging.NullHandler())

def check_disk_space(path=".", min_gb=1.0):
    """
    Ensures sufficient disk space exists on the VM.
    """
    try:
        total, used, free = shutil.disk_usage(path)
        free_gb = free / (1024**3)
        if free_gb < min_gb:
            logger.warning(f"DISK SPACE CRITICAL: Only {free_gb:.2f} GB free. Operations may fail.")
        else:
            logger.info(f"Disk Space Check: {free_gb:.2f} GB free. Proceeding.")
    except Exception as e:
        logger.error(f"Failed to check disk space: {e}")

def get_memory_limit():
    """
    Detects memory limit with Container Awareness.
    Checks Linux Cgroups v1/v2 for container limits before falling back to Host RAM.
    """
    # Try Cgroup V2
    try:
        with open("/sys/fs/cgroup/memory.max", "r") as f:
            mem_str = f.read().strip()
            if mem_str != "max":
                limit = int(mem_str)
                logger.info(f"Container Memory Limit Detected (Cgroup V2): {limit/1e9:.2f} GB")
                return limit
    except Exception:
        pass

    # Try Cgroup V1
    try:
        with open("/sys/fs/cgroup/memory/memory.limit_in_bytes", "r") as f:
            limit = int(f.read().strip())
            # Cgroup v1 often reports a huge number (approx 9e18) if unlimited
            if limit < 1e15: 
                logger.info(f"Container Memory Limit Detected (Cgroup V1): {limit/1e9:.2f} GB")
                return limit
    except Exception:
        pass

    # Fallback to Host Memory
    mem = psutil.virtual_memory().total
    logger.info(f"Host Memory Detected (psutil): {mem/1e9:.2f} GB")
    return mem

def calculate_optimal_batch_size(n_jobs=1, safety_factor=0.5, row_size_estimate=1024):
    """
    Calculates batch size dynamically based on available RAM and worker count.
    """
    available_mem = get_memory_limit()
    
    # Allocate memory per worker, applying safety factor
    mem_per_worker = (available_mem * safety_factor) / n_jobs
    
    estimated_rows = int(mem_per_worker / row_size_estimate)
    
    # Clamp between reasonable limits to prevent overhead or OOM
    optimal_batch = max(5000, min(estimated_rows, 500000))
    
    logger.info(f"Dynamic Batch Sizing: {available_mem/1e9:.2f}GB RAM / {n_jobs} Workers -> {optimal_batch} rows/batch")
    return optimal_batch

# --- Forensic & Data Structures ---

class BloomFilter:
    """
    A probabilistic data structure for efficient membership testing.
    Uses significantly less memory than a Python set for large datasets.
    """
    def __init__(self, n_items, p_false_positive=0.01):
        # Calculate optimal size (m) and hash count (k)
        self.m = int(-(n_items * math.log(p_false_positive)) / (math.log(2)**2))
        self.k = int((self.m / n_items) * math.log(2))
        
        # Initialise bit array (using bytearray for efficiency)
        self.bit_array = bytearray(self.m // 8 + 1)
        self.n_bits = self.m
        
        logger.info(f"Bloom Filter Initialised: {self.m} bits ({self.m/8/1024/1024:.2f} MB), {self.k} hashes.")

    def _hashes(self, item):
        """
        Generates k hashes using Double Hashing simulation.
        h(i) = (h1 + i * h2) % m
        """
        item_str = str(item).encode('utf-8')
        # Use MD5 and SHA1 as base hashes (standard lib, no extra deps)
        h1 = int(hashlib.md5(item_str).hexdigest(), 16)
        h2 = int(hashlib.sha1(item_str).hexdigest(), 16)
        
        for i in range(self.k):
            yield (h1 + i * h2) % self.n_bits

    def add(self, item):
        for bit_index in self._hashes(item):
            byte_index = bit_index // 8
            bit_offset = bit_index % 8
            self.bit_array[byte_index] |= (1 << bit_offset)

    def contains(self, item):
        for bit_index in self._hashes(item):
            byte_index = bit_index // 8
            bit_offset = bit_index % 8
            if not (self.bit_array[byte_index] & (1 << bit_offset)):
                return False
        return True

class StreamForensics:
    """
    Stateful forensic auditor for streaming data.
    Uses Bloom Filter to detect duplicates without loading full data.
    """
    def __init__(self, estimated_rows=10_000_000):
        self.bloom = BloomFilter(n_items=estimated_rows, p_false_positive=0.01)
        self.total_rows = 0
        self.dropped_rows = 0
        self.duplicate_count = 0
        
    def audit_chunk(self, df):
        """
        Audits a single chunk. Returns cleaned chunk.
        """
        initial_len = len(df)
        self.total_rows += initial_len
        
        # 1. Missing Values
        if df.isnull().values.any():
            df = df.dropna()
            
        # 2. Duplicates (Bloom Filter Check)
        keep_mask = []
        for idx in df.index:
            if self.bloom.contains(idx):
                keep_mask.append(False)
                self.duplicate_count += 1
            else:
                self.bloom.add(idx)
                keep_mask.append(True)
        
        df = df[keep_mask]
        
        dropped_in_chunk = initial_len - len(df)
        self.dropped_rows += dropped_in_chunk
        
        return df

    def log_summary(self):
        retention = ((self.total_rows - self.dropped_rows) / self.total_rows) * 100 if self.total_rows > 0 else 0
        logger.info(f"--- FORENSIC SUMMARY ---")
        logger.info(f"Total Rows Processed: {self.total_rows}")
        logger.info(f"Global Duplicates Dropped: {self.duplicate_count}")
        logger.info(f"Total Rows Dropped: {self.dropped_rows}")
        logger.info(f"Data Retention Rate: {retention:.2f}%")
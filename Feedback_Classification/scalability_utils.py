"""
Script Name: scalability_utils.py
Author: Shane Lee
Description: Shared Infrastructure Library. Provides container-aware resource detection
             by inspecting Linux Cgroups. Implements structured JSON logging
             and probabilistic data structures (Bloom Filters).
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

# --- Logging Utilities ---

class JSONFormatter(logging.Formatter):
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
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_filename)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers = []

    use_json = False
    if config and config.get("operational_settings", {}).get("enable_json_logging"):
        use_json = True

    if use_json:
        formatter = JSONFormatter()
    else:
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    file_handler = logging.FileHandler(log_path, mode='a')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger

def validate_config_version(config, expected_version="1.0"):
    version = config.get("schema_version", "0.0")
    if version != expected_version:
        error_msg = f"Config Version Mismatch: Expected {expected_version}, found {version}."
        print(f"CRITICAL: {error_msg}", file=sys.stderr)
        raise ValueError(error_msg)
    return True

# --- Resource Management Utilities ---

logger = logging.getLogger("ScalabilityUtils")
logger.addHandler(logging.NullHandler())

def get_memory_limit():
    """Detects memory limit with Container Awareness (Cgroups)."""
    try:
        with open("/sys/fs/cgroup/memory.max", "r") as f:
            mem_str = f.read().strip()
            if mem_str != "max":
                return int(mem_str)
    except Exception:
        pass
    try:
        with open("/sys/fs/cgroup/memory/memory.limit_in_bytes", "r") as f:
            limit = int(f.read().strip())
            if limit < 1e15: return limit
    except Exception:
        pass
    return psutil.virtual_memory().total

# --- Forensic & Data Structures ---

class BloomFilter:
    """Probabilistic data structure for O(1) deduplication."""
    def __init__(self, n_items, p_false_positive=0.01):
        self.m = int(-(n_items * math.log(p_false_positive)) / (math.log(2)**2))
        self.k = int((self.m / n_items) * math.log(2))
        self.bit_array = bytearray(self.m // 8 + 1)
        self.n_bits = self.m

    def _hashes(self, item):
        item_str = str(item).encode('utf-8')
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
    def __init__(self, estimated_rows=10_000_000):
        self.bloom = BloomFilter(n_items=estimated_rows, p_false_positive=0.01)
        self.total_rows = 0
        self.dropped_rows = 0
        
    def audit_chunk(self, df, id_col=None):
        initial_len = len(df)
        self.total_rows += initial_len
        
        if id_col and id_col in df.columns:
            keep_mask = []
            for val in df[id_col]:
                if self.bloom.contains(val):
                    keep_mask.append(False)
                else:
                    self.bloom.add(val)
                    keep_mask.append(True)
            df = df[keep_mask]
            
        self.dropped_rows += (initial_len - len(df))
        return df

    def log_summary(self):
        retention = ((self.total_rows - self.dropped_rows) / self.total_rows) * 100 if self.total_rows > 0 else 0
        logger.info(f"--- FORENSIC SUMMARY ---")
        logger.info(f"Total Rows: {self.total_rows} | Dropped: {self.dropped_rows} | Retention: {retention:.2f}%")
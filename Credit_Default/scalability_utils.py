"""Shared library for scalability and resource management.

Purpose:
    Provides shared utilities for configuration loading, logging, resource
    inspection, and streaming forensics used by this module.

Mechanisms:
    *   **Container Governance:** Cgroup V1/V2 memory limit detection.
    *   **Probabilistic Structures:** Bloom filter membership checks used by StreamForensics.
    *   **Observability:** JSON or text logging formatters.
    *   **Resilience:** Decorators for exponential backoff.

Author: Shane Lee
Licence: MIT
"""

import datetime
import os
import functools
import hashlib
import json
import logging
import math
import shutil
import sys
import time
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    List,
    Optional,
    TypeVar,
    Union,
)

import pandas as pd
import psutil

# --- Conditional Imports for Cloud Compatibility ---
try:
    from azure.ai.ml import MLClient
    from azure.identity import DefaultAzureCredential
except ImportError:
    # Required for orchestration (local), not for training (cloud container).
    MLClient = None  # type: ignore
    DefaultAzureCredential = None  # type: ignore


# --- Type Definitions ---
T = TypeVar("T")
PathLike = Union[str, Path]
ConfigType = Dict[str, Any]


# --- Configuration & Constants ---

class ProjectPaths:
    """Centralised registry of file paths and directory structures.

    Paths are resolved relative to the location of this script to improve
    stability across different execution contexts (local vs cloud).
    """

    _ROOT: Path = Path(__file__).resolve().parent

    # Data Directories
    INPUTS: Path = _ROOT / "inputs"
    OUTPUTS: Path = _ROOT / "outputs"
    LOGS: Path = _ROOT / "logs"
    ARTIFACTS: Path = _ROOT / "downloaded_artifacts"
    SRC: Path = _ROOT / "src"

    # Operational Files
    SCORING_INPUT: Path = INPUTS / "scoring_input_data.csv"
    DLQ: Path = INPUTS / "dlq_failed_chunks.csv"
    SCORING_OUTPUT: Path = OUTPUTS / "scoring_results_prediction.csv"

    # Audit & State
    LATEST_JOB: Path = LOGS / "latest_job.txt"
    FORENSIC_LOG: Path = LOGS / "forensic_audit.log"

    # Artifacts
    MODEL_FILENAME: str = "model.joblib"


# --- Logging Infrastructure ---

class JSONFormatter(logging.Formatter):
    """Formatter to output log records as machine-parseable JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        """Formats the log record as a JSON string.

        Args:
            record: The logging record.

        Returns:
            str: JSON formatted log string.
        """
        log_record: Dict[str, Any] = {
            "timestamp": datetime.datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "line": record.lineno,
        }
        if record.exc_info:
            log_record["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_record)


def configure_logging(
    name: str,
    log_filename: str,
    config: Optional[ConfigType] = None,
    log_to_file: bool = True
) -> logging.Logger:
    """Configures a standardised logger with console output and optional file output.

    Args:
        name: Name of the logger.
        log_filename: Filename for the log file stored in the resolved log directory.
        config: Optional configuration dict to determine formatting.
        log_to_file: Whether to write logs to disk.

    Returns:
        logging.Logger: Configured logger instance.
    """
    # Resolve Log Directory
    log_dir: Path = ProjectPaths.LOGS
    use_json: bool = False

    if config:
        ops_settings = config.get("operational_settings", {})
        config_log_dir = ops_settings.get("log_directory")
        if config_log_dir:
            log_dir = Path(config_log_dir)
        use_json = ops_settings.get("enable_json_logging", False)

    if log_to_file:
        log_dir.mkdir(parents=True, exist_ok=True)

    log_path: Path = log_dir / log_filename
    logger: logging.Logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)

    # Determine format
    formatter: Union[JSONFormatter, logging.Formatter] = (
        JSONFormatter() if use_json else logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    )

    # Console Handler
    console_handler: logging.StreamHandler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File Handler
    if log_to_file:
        file_handler: logging.FileHandler = logging.FileHandler(
            log_path, mode='a', encoding='utf-8'
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


# --- Resilience Decorators ---

def retry_with_backoff(
    retries: int = 3,
    backoff_in_seconds: int = 2,
    logger: Optional[logging.Logger] = None
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator to retry a function with exponential backoff.

    Args:
        retries: Maximum number of retries.
        backoff_in_seconds: Initial backoff duration.
        logger: Logger instance for reporting.

    Returns:
        Callable: Decorated function.
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            attempt: int = 0
            while True:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt >= retries:
                        if logger:
                            logger.error(
                                f"CRITICAL: Operation failed after {retries} retries. "
                                f"Error: {e}"
                            )
                        raise
                    
                    sleep_time: int = (backoff_in_seconds * 2 ** attempt)
                    if logger:
                        logger.warning(
                            f"Transient Error: {e}. Retrying in {sleep_time}s..."
                        )
                    
                    time.sleep(sleep_time)
                    attempt += 1
        return wrapper
    return decorator


# --- Configuration Management ---

def load_config(path: PathLike = "config.json") -> ConfigType:
    """Loads and validates the configuration file.

    Args:
        path: Path to the configuration JSON.

    Returns:
        ConfigType: Configuration dictionary.

    Raises:
        SystemExit: If config file is missing or malformed.
    """
    script_dir: Path = Path(__file__).parent
    config_path: Path = Path(path)

    if not config_path.is_absolute():
        config_path = script_dir / path

    if not config_path.exists():
        template_path: Path = config_path.with_name("config_template.json")
        sys.stderr.write(f"CRITICAL: Configuration file '{config_path}' missing.\n")
        if template_path.exists():
            sys.stderr.write(
                f"ACTION REQUIRED: Copy '{template_path.name}' to '{config_path.name}' "
                "and populate it.\n"
            )
        sys.exit(1)

    try:
        with config_path.open("r", encoding="utf-8") as f:
            config: ConfigType = json.load(f)
        _validate_config_version(config)
        return config
    except json.JSONDecodeError as e:
        sys.stderr.write(f"CRITICAL: Invalid JSON in '{config_path}': {e}\n")
        sys.exit(1)
    except Exception as e:
        sys.stderr.write(f"CRITICAL: Configuration error: {e}\n")
        sys.exit(1)


def _validate_config_version(
    config: ConfigType,
    expected_version: str = "1.1"
) -> None:
    """Internal validator for config schema version.

    Args:
        config: Loaded configuration dictionary.
        expected_version: The expected version string.
    """
    version: str = str(config.get("schema_version", "0.0"))
    if version != expected_version:
        logger: logging.Logger = logging.getLogger("ScalabilityUtils")
        logger.warning(f"Config Version Mismatch: Expected {expected_version}, found {version}")


def get_azure_client(config: ConfigType) -> Any:
    """Factory to create an authenticated MLClient.

    Args:
        config: Configuration dictionary containing Azure credentials.

    Returns:
        MLClient: Authenticated Azure ML client.

    Raises:
        SystemExit: If Azure SDK packages are missing.
    """
    if MLClient is None or DefaultAzureCredential is None:
        sys.stderr.write(
            "CRITICAL: Azure SDK not installed. "
            "Run 'pip install azure-ai-ml azure-identity'.\n"
        )
        sys.exit(1)

    try:
        credential: DefaultAzureCredential = DefaultAzureCredential()
        return MLClient(
            credential=credential,
            subscription_id=config["subscription_id"],
            resource_group_name=config["resource_group"],
            workspace_name=config["workspace_name"],
        )
    except Exception as e:
        sys.stderr.write(f"CRITICAL: Azure Client Instantiation Failed: {e}\n")
        sys.exit(1)


# --- State Management ---

def get_latest_job_id(log_dir: PathLike = ProjectPaths.LOGS) -> Optional[str]:
    """Retrieves the ID of the most recently submitted job.

    Args:
        log_dir: Directory containing the state file.

    Returns:
        Optional[str]: The job ID if found, else None.
    """
    path: Path = Path(log_dir) / "latest_job.txt"
    if path.exists():
        return path.read_text(encoding="utf-8").strip()
    return None


def save_job_id(job_id: str, log_dir: PathLike = ProjectPaths.LOGS) -> None:
    """Saves the ID of the current job to the state file.

    Args:
        job_id: The job identifier to save.
        log_dir: Directory where the state file is stored.
    """
    path: Path = Path(log_dir)
    path.mkdir(parents=True, exist_ok=True)
    (path / "latest_job.txt").write_text(job_id, encoding="utf-8")


# --- Data Validation ---

def validate_dataframe_schema(
    df: pd.DataFrame,
    schema_config: Dict[str, Any],
    logger: Optional[logging.Logger] = None
) -> pd.DataFrame:
    """Validates required columns and attempts dtype casting per configuration.

    Args:
        df: Input DataFrame.
        schema_config: Schema definition from config.
        logger: Logger for warnings/errors.

    Returns:
        pd.DataFrame: Validated (and potentially type-cast) DataFrame.

    Raises:
        ValueError: If required columns are missing.
    """
    if not schema_config:
        return df

    required_cols: List[str] = schema_config.get("required_columns", [])
    missing_cols: List[str] = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        error_msg: str = f"Schema Violation: Missing columns {missing_cols}"
        if logger:
            logger.error(error_msg)
        raise ValueError(error_msg)

    type_expectations: Dict[str, str] = schema_config.get("dtypes", {})
    for col, dtype in type_expectations.items():
        if col in df.columns:
            try:
                df[col] = df[col].astype(dtype)
            except Exception as e:
                warning_msg: str = (
                    f"Schema Warning: Column {col} could not be cast to {dtype}. "
                    f"Error: {e}"
                )
                if logger:
                    logger.warning(warning_msg)

    return df


# --- Resource Management ---

def check_disk_space(path: PathLike = ".", min_gb: float = 1.0) -> None:
    """Checks available disk space and logs warnings when below the threshold.

    Args:
        path: Path to check.
        min_gb: Minimum required space in Gigabytes.
    """
    logger: logging.Logger = logging.getLogger("ScalabilityUtils")
    try:
        total, used, free = shutil.disk_usage(path)
        free_gb: float = free / (1024**3)
        if free_gb < min_gb:
            logger.warning(f"DISK SPACE CRITICAL: Only {free_gb:.2f} GB free.")
        else:
            logger.info(f"Disk Space Check: {free_gb:.2f} GB free.")
    except Exception as e:
        logger.error(f"Failed to check disk space: {e}")


def get_memory_limit() -> int:
    """Detects memory limit with Container Awareness (Cgroups V1/V2).

    Falls back to Host RAM if not in a container.

    Returns:
        int: Memory limit in bytes.
    """
    logger: logging.Logger = logging.getLogger("ScalabilityUtils")

    # Try Cgroup V2
    try:
        cgroup2: Path = Path(os.sep) / "sys" / "fs" / "cgroup" / "memory.max"
        if cgroup2.exists():
            content: str = cgroup2.read_text().strip()
            if content != "max":
                limit: int = int(content)
                logger.info(f"Container Memory Limit (Cgroup V2): {limit/1e9:.2f} GB")
                return limit
    except Exception:
        pass

    # Try Cgroup V1
    try:
        cgroup1: Path = Path(os.sep) / "sys" / "fs" / "cgroup" / "memory" / "memory.limit_in_bytes"
        if cgroup1.exists():
            limit: int = int(cgroup1.read_text().strip())
            # Filter out very large "unlimited" values
            if limit < 1e15:
                logger.info(f"Container Memory Limit (Cgroup V1): {limit/1e9:.2f} GB")
                return limit
    except Exception:
        pass

    # Fallback to Host Memory
    try:
        mem: int = psutil.virtual_memory().total
        logger.info(f"Host Memory Detected: {mem/1e9:.2f} GB")
        return mem
    except Exception as e:
        logger.warning(f"Could not determine memory limit: {e}. Defaulting to 1GB.")
        return 1024 * 1024 * 1024


def calculate_optimal_batch_size(
    n_jobs: int = 1,
    safety_factor: float = 0.5,
    row_size_estimate_bytes: int = 1024
) -> int:
    """Calculates batch size dynamically based on available RAM.

    Args:
        n_jobs: Number of parallel workers.
        safety_factor: Fraction of memory to allocate (0.0 - 1.0).
        row_size_estimate_bytes: Estimated bytes per row.

    Returns:
        int: Optimal number of rows per batch.
    """
    logger: logging.Logger = logging.getLogger("ScalabilityUtils")
    available_mem: int = get_memory_limit()

    # Distribute memory across workers
    mem_per_worker: float = (available_mem * safety_factor) / max(1, n_jobs)
    estimated_rows: int = int(mem_per_worker / row_size_estimate_bytes)

    # Clamp to reasonable limits to bound batch size
    # Min: 5000 rows, Max: 500,000 rows
    optimal_batch: int = max(5000, min(estimated_rows, 500000))

    logger.info(
        f"Dynamic Batch Sizing: {available_mem/1e9:.2f}GB RAM / {n_jobs} Workers "
        f"-> {optimal_batch} rows/batch"
    )
    return optimal_batch


def resolve_batch_size(
    config: ConfigType,
    n_jobs: int = 1,
    row_size_estimate_bytes: int = 1024
) -> int:
    """Determines batch size: Configured Value > Dynamic Calculation.

    Args:
        config: Configuration dictionary.
        n_jobs: Number of parallel workers.
        row_size_estimate_bytes: Estimated bytes per row.

    Returns:
        int: The resolved batch size.
    """
    logger: logging.Logger = logging.getLogger("ScalabilityUtils")
    ops_settings: Dict[str, Any] = config.get("operational_settings", {})
    config_batch: Optional[int] = ops_settings.get("batch_size")

    if config_batch:
        logger.info(f"Using Configured Batch Size: {config_batch}")
        return int(config_batch)

    return calculate_optimal_batch_size(
        n_jobs=n_jobs,
        safety_factor=0.5,
        row_size_estimate_bytes=row_size_estimate_bytes
    )


def find_model_artifact(base_path: PathLike) -> Optional[Path]:
    """Recursively searches for the model artifact.

    Args:
        base_path: The directory to search within.

    Returns:
        Optional[Path]: The path to the model file if found.
    """
    search_path: Path = Path(base_path)
    if not search_path.exists():
        return None

    # rglob returns a generator; next() gets the first match safely
    return next(search_path.rglob(ProjectPaths.MODEL_FILENAME), None)


# --- Forensic & Data Structures ---

class BloomFilter:
    """A probabilistic data structure for efficient membership testing (O(m) space)."""

    def __init__(self, n_items: int, p_false_positive: float = 0.01) -> None:
        """Initialises the Bloom Filter.

        Args:
            n_items: Estimated number of items to add.
            p_false_positive: Desired false positive probability.
        """
        # Optimal size (m)
        self.m: int = int(-(n_items * math.log(p_false_positive)) / (math.log(2)**2))
        # Optimal hash functions (k)
        self.k: int = int((self.m / n_items) * math.log(2))

        self.bit_array: bytearray = bytearray((self.m // 8) + 1)
        self.n_bits: int = self.m

        self.logger: logging.Logger = logging.getLogger("ScalabilityUtils")
        self.logger.info(f"Bloom Filter Initialised: {self.m} bits, {self.k} hashes.")

    def _hashes(self, item: Any) -> Generator[int, None, None]:
        """Generates k hashes using Double Hashing.

        Args:
            item: The item to hash.

        Yields:
            int: Bit index.
        """
        item_str: bytes = str(item).encode('utf-8')
        # Use hashlib for consistent cross-platform hashing
        h1: int = int(hashlib.md5(item_str).hexdigest(), 16)
        h2: int = int(hashlib.sha1(item_str).hexdigest(), 16)

        for i in range(self.k):
            yield (h1 + i * h2) % self.n_bits

    def add(self, item: Any) -> None:
        """Adds an item to the filter.

        Args:
            item: The item to add.
        """
        for bit_index in self._hashes(item):
            byte_index: int = bit_index // 8
            bit_offset: int = bit_index % 8
            self.bit_array[byte_index] |= (1 << bit_offset)

    def contains(self, item: Any) -> bool:
        """Checks if an item is likely in the filter.

        Args:
            item: The item to check.

        Returns:
            bool: True if the item is in the filter (with possible false positive).
        """
        for bit_index in self._hashes(item):
            byte_index: int = bit_index // 8
            bit_offset: int = bit_index % 8
            if not (self.bit_array[byte_index] & (1 << bit_offset)):
                return False
        return True


class StreamForensics:
    """Stateful forensic auditor for streaming data.

    Tracks data retention, duplicates, and drop rates across ingestion batches.
    """

    def __init__(self, estimated_rows: int = 10_000_000) -> None:
        """Initialises the forensics module.

        Args:
            estimated_rows: Expected volume for Bloom Filter sizing.
        """
        self.bloom: BloomFilter = BloomFilter(n_items=estimated_rows, p_false_positive=0.01)
        self.total_rows: int = 0
        self.dropped_rows: int = 0
        self.duplicate_count: int = 0
        self.logger: logging.Logger = logging.getLogger("ScalabilityUtils")

    def audit_chunk(self, df: pd.DataFrame) -> pd.DataFrame:
        """Audits a data chunk, removing duplicates and nulls.

        Args:
            df: Input DataFrame chunk.

        Returns:
            pd.DataFrame: Cleaned chunk.
        """
        initial_len: int = len(df)
        self.total_rows += initial_len

        # 1. Null Safety
        if df.isnull().values.any():
            df = df.dropna()

        # 2. Deduplication (Bloom Filter)
        keep_mask: List[bool] = []
        for idx in df.index:
            if self.bloom.contains(idx):
                self.duplicate_count += 1
                keep_mask.append(False)
            else:
                self.bloom.add(idx)
                keep_mask.append(True)

        df = df[keep_mask]

        self.dropped_rows += (initial_len - len(df))
        return df

    def log_summary(self) -> None:
        """Logs the final retention statistics."""
        retention: float = 0.0
        if self.total_rows > 0:
            retention = (
                (self.total_rows - self.dropped_rows) / self.total_rows
            ) * 100

        self.logger.info("--- FORENSIC SUMMARY ---")
        self.logger.info(f"Total Rows Processed: {self.total_rows}")
        self.logger.info(f"Global Duplicates Dropped: {self.duplicate_count}")
        self.logger.info(f"Total Rows Dropped: {self.dropped_rows}")
        self.logger.info(f"Data Retention Rate: {retention:.2f}%")

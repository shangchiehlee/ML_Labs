# Credit Default Prediction Module

**System Owner:** Shane Lee
**Licence:** MIT

---

## 1. Executive Summary

This module implements an **Ensemble Machine Learning** architecture designed to predict credit default probabilities. The system decouples computational workloads by executing model training on Azure Compute Clusters while performing inference on local infrastructure.

The architecture supports the processing of datasets exceeding available memory ($N > 10^7$) through **incremental learning**. It employs container-aware resource management, configurable batch processing, **Kernel Approximation**, and **Streaming Ensembles** to maintain constant memory usage during execution.

**Operational Capabilities:**
*   **Scalability:** The system processes data in discrete batches. Memory usage is determined by the batch size and remains constant ($O(1)$) regardless of the total dataset size.
*   **Resource Efficiency:** The system utilises `partial_fit` operations. This allows execution on hardware with limited RAM without triggering Out-Of-Memory (OOM) errors.
*   **Resilience:** The pipeline includes automated integrity checks and a Dead Letter Queue (DLQ) to isolate malformed data without halting execution.

---

## 2. Technical Architecture

### Design Patterns
The system utilises a **Streaming Ensemble** pattern augmented with **Nystroem Kernel Approximation**.
*   **Feature Projection:** The module projects features into a higher-dimensional space (1200 components via `sklearn.kernel_approximation.Nystroem`) to allow linear models to approximate non-linear decision boundaries.
*   **Ensemble Aggregation:** The system trains 7 independent Stochastic Gradient Descent (SGD) classifiers. During inference, predictions are aggregated via soft voting (averaging probabilities).

### Algorithmic Complexity
*   **Space Complexity:** **Constant ($O(1)$)**. The memory footprint is constrained by the configured `batch_size` and model parameters. The system utilises `pd.read_csv(chunksize=...)` and `model.partial_fit()` to process data without loading the full dataset into RAM.
*   **Time Complexity:** **Linear ($O(N)$)**. The system performs a single pass over the training data (per epoch) and a single pass over the inference data.

### Resource Governance
The module includes a library (`scalability_utils.py`) for **Container-Aware Resource Management**.
*   **Cgroup Inspection:** The system inspects Linux Control Groups v1/v2 (`/sys/fs/cgroup/memory.max`) to identify memory limits within containerised environments (e.g. Azure ML, Docker). This overrides standard Host OS memory reporting.
*   **Dynamic Batch Sizing:** If a batch size is not explicitly configured, the system calculates a size based on the detected available memory and a safety factor.
*   **Memory Mapping:** During inference, the model artifact is loaded using `mmap_mode='r'`. This allows multiple worker processes to share physical memory pages for the read-only model object.

---

## 3. System Integrity & Reliability

### Data Integrity
*   **Bloom Filters:** The system utilises a probabilistic **Bloom Filter** to detect duplicate records in data streams with a fixed memory footprint.
*   **Schema Validation:** The system verifies column presence and data types against the contract defined in `config.json` before processing.

### Fault Tolerance
*   **Dead Letter Queue (DLQ):** The ETL pipeline (`03_generate_input_data.py`) routes malformed chunks or schema violations to a separate error file (`inputs/dlq_failed_chunks.csv`). This allows the pipeline to proceed with valid data.
*   **Asynchronous I/O:** The data generation script uses a `ThreadPoolExecutor` to write processed chunks to disk asynchronously.
*   **Zero-Byte Detection:** Artifact retrieval scripts enforce zero-byte checks. Transfers resulting in empty files trigger a process halt.

---

## 4. Directory Manifest

```text
Credit_Default/
├── inputs/                 <-- Staging: Structured data for inference and DLQ files
├── outputs/                <-- Results: Scored data, predictions, and forensic logs
├── logs/                   <-- Audit: Operational logs and state files
│   ├── job_history.csv     <-- Ledger: Record of job submissions
│   ├── latest_job.txt      <-- State: ID of the most recent job
│   └── *.log               <-- Logs: Execution logs
├── src/
│   └── main.py             <-- Training Script: Executed on Azure Compute
├── scalability_utils.py    <-- Shared Library: Container-aware resource management logic
├── 01_submit_job.py        <-- Orchestrator: Submits workload to Azure
├── 02_download_outputs.py  <-- Retrieval: Downloads artifacts with integrity checks
├── 03_generate_input_data.py <-- ETL: Generates/Downloads input data via Async I/O
├── 04_local_model_scoring.py <-- Inference: Local batch scoring engine
├── 05_archive_job.py       <-- Lifecycle: Soft-deletes Azure jobs
├── 06_cleanup_local.py     <-- Hygiene: Removes temporary local artifacts
└── config.json             <-- Configuration: Credentials, hyperparameters, and schema
```

---

## 5. Standard Operating Procedure (SOP)

### Phase 1: Configuration & Training
1.  **Configuration:** Populate `config.json` with Azure credentials, hyperparameters, and schema definitions. Set `batch_size` in `operational_settings`.
2.  **Submission:** Execute `python 01_submit_job.py` to serialise the context, package shared utilities, and submit the job to the Azure Compute Cluster.
3.  **Retrieval:** Execute `python 02_download_outputs.py` to retrieve the artifact and perform integrity checks.

### Phase 2: Local Operations
1.  **Data Generation:** Execute `python 03_generate_input_data.py` to stream the raw dataset.
    *   *Configuration:* Set `"dry_run_rows": 1000` in `config.json` to limit processing for testing.
2.  **Inference:** Execute `python 04_local_model_scoring.py` to initiate batch processing and generate predictions in `outputs/`.

### Phase 3: Lifecycle Management
1.  **Archival:** Execute `python 05_archive_job.py` to soft-delete the job in Azure ML Studio.
2.  **Sanitisation:** Execute `python 06_cleanup_local.py` to remove temporary artifacts, logs, and build copies of shared utilities.
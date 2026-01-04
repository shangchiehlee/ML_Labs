# Graduate Admissions Assessment Module

**System Owner:** Shane Lee
**Licence:** MIT

## Executive Summary

This module implements a **Stochastic Gradient Descent (SGD) Regression** pipeline designed to estimate admission probabilities based on quantitative metrics (GRE, TOEFL, CGPA).

The system utilises an **Incremental Learning** workflow. By employing the `partial_fit` method within the `SGDRegressor` algorithm, the application processes datasets in discrete batches. This decouples memory consumption from the total dataset size, allowing the processing of datasets that exceed available Random Access Memory (RAM) without requiring vertical infrastructure scaling.

## Technical Architecture

### 1. Core Design Pattern: Incremental Learning
The training script (`main.py`) implements an **Online Learning** paradigm.

*   **Mechanism:** Data is ingested via `pd.read_csv` with a defined `chunksize`. The model weights are updated iteratively using `SGDRegressor.partial_fit`.
*   **Complexity:** The system maintains **Constant Memory Complexity ($O(1)$)** relative to the total row count. Memory usage is bounded by the batch size and model parameters.
*   **Batch Sizing:** The batch size is calculated dynamically at runtime via `scalability_utils.calculate_optimal_batch_size` to align with available resources.

### 2. Resource Governance
The module uses `scalability_utils.py` to detect container-specific memory limits rather than host-level metrics.

*   **Detection Logic:** The system inspects Linux Control Groups (cgroups) via `/sys/fs/cgroup/memory.max` (v2) or `/sys/fs/cgroup/memory/memory.limit_in_bytes` (v1).
*   **Application:** If a cgroup limit is detected, the application uses this value to constrain buffer sizes; otherwise, it defaults to `psutil` host memory readings.

### 3. Stream Deduplication
To detect duplicate records in a streaming context without linear memory growth ($O(N)$), the system implements a **Bloom Filter**.

*   **Implementation:** The `BloomFilter` class uses a bit array and double hashing (MD5/SHA1) to test for set membership.
*   **Constraint:** This provides probabilistic deduplication with fixed memory usage, independent of the number of processed records.

## System Integrity & Reliability

### 1. Fault Tolerance
*   **Dead Letter Queue (DLQ):** The ETL script (`03_generate_input_data.py`) isolates records that fail processing into a separate file (`inputs/dlq_failed_chunks.csv`) rather than terminating the process.
*   **Exponential Backoff:** Network requests in the data generation script are wrapped in a `retry_with_backoff` decorator, which implements increasing delays between failed attempts.

### 2. Artifact Verification
*   **Manifest Generation:** The retrieval script (`02_download_outputs.py`) generates a JSON manifest (`download_manifest.json`) listing downloaded files and their timestamps.
*   **Integrity Check:** The script validates that downloaded artifacts have a file size greater than zero bytes. Zero-byte files trigger a warning and are flagged in the manifest.

### 3. Concurrency
*   **I/O Operations:** The data generation script uses `ThreadPoolExecutor` to perform disk writes asynchronously, preventing blocking on the main ingestion thread.
*   **Inference:** The scoring engine (`04_local_model_scoring.py`) utilises `ProcessPoolExecutor` to distribute batch predictions across CPU cores. It loads the model using `joblib` with `mmap_mode='r'` to share memory pages across worker processes.

## Directory Manifest

```text
Graduate_Admissions/
│
├── inputs/                    <-- Staging: Structured data and DLQ output
├── outputs/                   <-- Reporting: Prediction results and logs
├── logs/                      <-- Audit: Execution history and state files
├── src/                       <-- Logic: Source code for cloud execution
│   ├── main.py                <-- Entry Point: SGD training script
│   └── scalability_utils.py   <-- Library: Resource management utilities
│
├── scalability_utils.py       <-- Library: Local copy for pre-flight checks
├── 01_submit_job.py           <-- Orchestrator: Azure ML job submission
├── 02_download_outputs.py     <-- Collector: Artifact retrieval and verification
├── 03_generate_input_data.py  <-- Transformer: Async ETL and validation
├── 04_local_model_scoring.py  <-- Inference: Parallel batch scoring
├── 05_archive_job.py          <-- Lifecycle: Job archival
├── 06_cleanup_local.py        <-- Utility: Workspace sanitisation
│
├── config.json                <-- Configuration: Credentials and parameters
└── README.md                  <-- Documentation: System overview
```

## Standard Operating Procedure (SOP)

### Phase 1: Configuration
1.  **Credentials:** Populate `config.json` with the Azure Subscription ID, Resource Group, and Workspace Name.
2.  **Parameters:** Set `operational_settings.batch_size` in `config.json` based on available hardware.

### Phase 2: Training (Cloud)
1.  **Submission:** Run `python 01_submit_job.py`. This packages the `src` directory and submits the command job to Azure.
2.  **Retrieval:** Upon job completion, run `python 02_download_outputs.py` to download `model.joblib` and logs to `downloaded_artifacts/`.

### Phase 3: Inference (Local)
1.  **Data Preparation:** Run `python 03_generate_input_data.py`. This downloads raw data, applies schema rules, and routes failures to the DLQ.
2.  **Scoring:** Run `python 04_local_model_scoring.py`. This executes parallel batch predictions and outputs to `outputs/scoring_results_prediction.csv`.

### Phase 4: Maintenance
1.  **Archival:** Run `python 05_archive_job.py` to archive the Azure job entity.
2.  **Cleanup:** Run `python 06_cleanup_local.py` to delete local data directories (`inputs`, `outputs`, `logs`) and temporary artifacts.
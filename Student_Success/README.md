# Student Success & Retention Module

**System Owner:** Shane Lee
**Licence:** MIT

***

## 1. Executive Summary

This module implements a **Multi-Class Classification System** designed to predict student outcomes (Dropout, Enrolled, Graduate).

The system architecture decouples memory consumption from dataset size. By utilising **Out-of-Core Learning**, the pipeline processes data in streams rather than loading the entire dataset into Random Access Memory (RAM). This design enables the training of models on datasets that exceed the physical memory limits of the host infrastructure.

The workflow is hybrid. Training executes on Azure Compute Clusters to leverage scalable cloud resources, while inference (scoring) is optimised for local execution using memory mapping and parallel processing.

***

## 2. Technical Architecture

### 2.1. Core Design Pattern: Incremental Learning
The training pipeline (`main.py`) implements the `SGDClassifier` from the Scikit-Learn library. It utilises the `partial_fit()` method to update model parameters incrementally.

*   **Mechanism:** The script iterates through the source CSV in configurable batches (defined by `batch_size` in `config.json`).
*   **State Management:** Weights and intercept parameters are updated per batch. The system performs a pre-scan of the dataset to identify all unique target classes before training commences.
*   **Algorithmic Complexity:** The system maintains **Constant Memory Complexity ($O(1)$)** relative to the total number of input rows. Memory usage is strictly determined by the batch size and feature dimensionality, not the total dataset size.

### 2.2. Resource Governance
The system includes a dedicated library (`scalability_utils.py`) for infrastructure-aware resource management.

*   **Container Awareness:** The library inspects Linux Control Groups (v1 and v2) via `/sys/fs/cgroup/memory.max` to determine the configured memory limits of the containerised environment.
*   **Memory Mapping:** The inference engine (`04_local_model_scoring.py`) loads the serialised model using `joblib` with `mmap_mode='r'`. This maps the model file to virtual memory, allowing multiple worker processes to share physical memory pages and reducing the resident set size (RSS).
*   **Optimised Serialisation:** The training script forces `compress=0` during model persistence. This is a technical prerequisite for enabling memory mapping during the inference phase.

### 2.3. Inference Mechanics
The scoring engine operates independently of the training pipeline.

1.  **Parallel Execution:** A `ProcessPoolExecutor` distributes data chunks across available CPU cores.
2.  **Probabilistic Output:** The classifier uses the 'log_loss' function to output a probability distribution across the target classes.
3.  **Confidence Scoring:** The system calculates a **Prediction Confidence** score, defined as the maximum probability within the predicted class distribution.

***

## 3. System Integrity & Reliability

### 3.1. Probabilistic Deduplication
The system implements a **Bloom Filter** in `scalability_utils.py`. This probabilistic data structure tests for record existence using a fixed memory allocation ($O(1)$). It allows the pipeline to identify and exclude duplicate records during the training stream without maintaining a hash map of every seen record.

### 3.2. Fault Tolerance (Dead Letter Queue)
The Data Preparation script (`03_generate_input_data.py`) includes a **Dead Letter Queue (DLQ)** mechanism.
*   **Validation:** The script attempts to parse each data chunk.
*   **Isolation:** If parsing fails (e.g. due to malformed CSV rows), the specific chunk is written to `inputs/dlq_failed_chunks.csv`. This prevents a single corrupt batch from halting the entire ETL process.

### 3.3. Artifact Verification
The retrieval script (`02_download_outputs.py`) executes a file size check on downloaded artifacts. It generates a `download_manifest.json` listing the file inventory and explicitly flags any zero-byte files as corrupt.

***

## 4. Directory Manifest

```text
Student_Success/
│
├── inputs/                    <-- Staging: Structured data for scoring & DLQ
├── outputs/                   <-- Reporting: Prediction results & forensic logs
├── logs/                      <-- Audit: Operational logs and state files
├── src/                       <-- Logic: main.py (Incremental Training Script)
│
├── scalability_utils.py       <-- Library: Cgroup detection & Bloom Filter logic
├── 01_submit_job.py           <-- Orchestrator: Submits workload to Azure Compute
├── 02_download_outputs.py     <-- Collector: Retrieves artifacts with integrity checks
├── 03_generate_input_data.py  <-- Transformer: ETL with DLQ and Async I/O
├── 04_local_model_scoring.py  <-- Inference: Parallel batch scoring
├── 05_archive_job.py          <-- Lifecycle: Archives Azure jobs
├── 06_cleanup_local.py        <-- Utility: Workspace sanitisation
│
├── config.json                <-- Configuration: Batch sizes & Azure details
└── job_history.csv            <-- Ledger: Local record of job submissions
```

***

## 5. Standard Operating Procedure (SOP)

### Phase 1: Cloud Training

1.  **Configuration:** Update `config.json` with valid Azure credentials, subscription IDs, and the target `batch_size`.
2.  **Submission:** Execute `python 01_submit_job.py`. This script packages the `src` directory, injects the configuration, and submits the job to Azure Machine Learning.
3.  **Retrieval:** Execute `python 02_download_outputs.py`. This downloads the `model.joblib` and forensic logs, performing an immediate integrity check.

### Phase 2: Local Inference

4.  **Data Preparation:** Execute `python 03_generate_input_data.py`. This streams raw data, removes the target column (blind prediction), and routes failures to the DLQ.
5.  **Scoring:** Execute `python 04_local_model_scoring.py`. This loads the model via memory mapping and appends predictions to `outputs/scoring_results_prediction.csv`.

### Phase 3: Maintenance

6.  **Archival:** Execute `python 05_archive_job.py` to archive the Azure job. Note that jobs using Azure Registry Environments may have archival limitations.
7.  **Sanitisation:** Execute `python 06_cleanup_local.py` to force-delete local data directories (`inputs/`, `outputs/`) and logs. This script handles file permission errors associated with Windows file locking.
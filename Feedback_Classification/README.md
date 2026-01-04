# Feedback Classification Module

**System Owner:** Shane Lee
**Licence:** MIT

***

## 1. Executive Summary

This module implements a binary classification system designed to process streams of unstructured text data.

The system addresses the technical constraint of processing datasets that exceed available physical memory (RAM). By utilising **Out-of-Core Learning**, the application processes data in discrete batches. This architecture decouples memory consumption from the total volume of the dataset.

Consequently, the system maintains a fixed memory footprint regardless of input size. This enables the execution of training and inference workloads on memory-constrained compute environments.

***

## 2. Technical Architecture

### 2.1. Algorithmic Design
The system employs **Incremental Learning** to update model parameters sequentially. This approach negates the requirement to load the entire historical dataset into memory simultaneously.

*   **Stateless Feature Extraction:** The system utilises a `HashingVectorizer`. This component maps text tokens to feature indices using a MurmurHash3 function. This technique eliminates the need for an in-memory vocabulary dictionary, resulting in **Constant Space Complexity ($O(1)$)** relative to the diversity of the input text.
*   **Online Classification:** The `PassiveAggressiveClassifier` updates internal weights via the `partial_fit` method. This allows the model to adjust to new data points without retraining from scratch.

### 2.2. Resource Governance
The system implements active resource monitoring via the `scalability_utils` library to prevent Out-of-Memory (OOM) errors.

*   **Container Awareness:** The application inspects Linux Control Groups (Cgroups) at `/sys/fs/cgroup/memory.max` to determine the hard memory limits of the hosting container.
*   **Batch Processing:** Data ingestion is controlled via a configurable `batch_size` parameter. This ensures that the working set of data never exceeds the detected memory ceiling.

### 2.3. Inference Mechanics
The scoring engine (`04_local_model_scoring.py`) executes predictions using a parallel processing pipeline:

1.  **Memory Mapping:** The model artifact is loaded using `joblib` with `mmap_mode='r'`. This instructs the operating system to map the file directly into virtual memory, allowing multiple worker processes to share the same physical memory pages.
2.  **Parallel Execution:** A `ProcessPoolExecutor` distributes data chunks across available CPU cores to maximise throughput.
3.  **Confidence Scoring:** The system calculates a confidence metric based on the absolute distance of the data point from the decision hyperplane.

***

## 3. System Integrity & Reliability

### 3.1. Data Integrity
*   **Probabilistic Deduplication:** The system utilises **Bloom Filters** to identify duplicate records within the stream. This data structure supports membership testing with constant space complexity ($O(1)$).
*   **Artifact Verification:** The retrieval subsystem (`02_download_outputs.py`) performs zero-byte integrity checks on downloaded files to verify that data transfer completed successfully.

### 3.2. Fault Tolerance
*   **Dead Letter Queue (DLQ):** The ETL pipeline (`03_generate_input_data.py`) implements a DLQ pattern. Malformed data chunks that cause processing errors are isolated and written to `inputs/dlq_failed_chunks.csv`. This prevents a single corrupt batch from halting the entire pipeline.
*   **Output Validation:** The inference engine enforces strict type casting on output columns to ensure downstream compatibility.

***

## 4. Directory Manifest

```text
Feedback_Classification/
│
├── inputs/                    <-- Staging: Raw data and generated blind datasets
├── outputs/                   <-- Reporting: System artifacts and predictions
├── logs/                      <-- Audit: Operational logs and state files
├── src/                       <-- Logic: Cloud training scripts
│   └── main.py                <-- Core Logic: Incremental Training and Reporting
│
├── scalability_utils.py       <-- Shared Library: Resource detection and logging
├── 01_submit_job.py           <-- Orchestrator: Submits workload to Azure Compute
├── 02_download_outputs.py     <-- Collector: Retrieves and publishes artifacts
├── 03_generate_input_data.py  <-- Transformer: ETL with Dead Letter Queue
├── 04_local_model_scoring.py  <-- Inference: Parallel batch scoring engine
├── 05_archive_job.py          <-- Lifecycle: Archives Azure jobs
├── 06_cleanup_local.py        <-- Utility: Workspace sanitisation
│
├── config.json                <-- Configuration: Credentials and Hyperparameters
└── README.md                  <-- Documentation: System architecture and SOP
```

***

## 5. Standard Operating Procedure (SOP)

### Phase 1: Cloud Training
1.  **Configuration:** Populate `config.json` with valid Azure credentials and compute target details.
2.  **Submission:** Execute `python 01_submit_job.py`. This script packages the source code and dispatches the training workload to the cloud environment.
3.  **Retrieval:** Execute `python 02_download_outputs.py`. This retrieves the trained model artifact and publishes the following reports to the `outputs/` directory:
    *   `priority_alerts.csv` (High-confidence negative predictions)
    *   `sentiment_drivers.csv` (Keyword frequency analysis)
    *   `forensic_audit_full.csv` (Test set performance log)

### Phase 2: Local Inference
4.  **Data Preparation:** Execute `python 03_generate_input_data.py`. This script transforms raw data into a format suitable for scoring by removing target variables.
5.  **Execution:** Execute `python 04_local_model_scoring.py`. This generates predictions on the prepared data. Results are saved to `outputs/scoring_results_prediction.csv`.

### Phase 3: Lifecycle Management
6.  **Archival:** Execute `python 05_archive_job.py` to archive the job within the Azure workspace.
7.  **Sanitisation:** Execute `python 06_cleanup_local.py` to remove temporary artifacts, logs, and generated data from the local environment.
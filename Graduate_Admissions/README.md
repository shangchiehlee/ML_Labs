# Graduate Admissions Assessment Module: Incremental Predictive Pipeline

**Author:** Shane Lee  
**Licence:** MIT  

---

## 1. System Overview

**Operational Capability**  
This module performs incremental training and local batch scoring for graduate admissions prediction. Training reads CSV input in chunks, normalises headers, removes the "Serial No." column when present, and updates a StandardScaler and SGDRegressor using partial_fit (Source: `src/main.py`, `train_epoch`). When evaluation data is accumulated on the final epoch, the module writes `outputs/model.joblib` and logs performance metrics (Source: `src/main.py`, `main`). ETL streams from the configured data source, drops the target column for blind inference, and writes `scoring_input_data.csv` while appending failed chunks to `dlq_failed_chunks.csv` (Source: `03_generate_input_data.py`, `execute_etl_stream`). Local scoring uses ProcessPoolExecutor and writes `scoring_results_prediction.csv` (Source: `04_local_model_scoring.py`, `main`).

**Technical Implementation**  
The training worker uses StandardScaler and SGDRegressor with learning_rate="adaptive" and partial_fit on streaming batches (Source: `src/main.py`, `main`). Metrics are derived from the final epoch predictions and include regression metrics (R2, MSE, MAE) and classification metrics (accuracy, F1, precision, recall), with AUC recorded only when both classes are present (Source: `src/main.py`, `calculate_metrics`).

*   **Incremental Learning:** Uses StandardScaler.partial_fit and SGDRegressor.partial_fit on streaming batches (Source: `src/main.py`, `train_epoch`).
*   **Memory Profile:** Chunked ingestion bounds per-batch memory. Final epoch evaluation buffers scale with the number of retained test rows (Source: `src/main.py`).
*   **Input Traversal:** Each epoch iterates over the input stream in chunks; local scoring iterates over the scoring input in chunks (Source: `src/main.py`, `train_epoch`; `04_local_model_scoring.py`, `main`).
*   **Resource Governance:** `resolve_batch_size` uses the configured batch size when present; otherwise it derives a batch size from `get_memory_limit` (cgroup V1/V2 when available) for ETL and scoring (Source: `scalability_utils.py`, `resolve_batch_size`, `get_memory_limit`; `03_generate_input_data.py`, `04_local_model_scoring.py`).
*   **Stream Auditing:** StreamForensics drops nulls and removes repeated indices using a Bloom filter during the first epoch, while ETL routes failed chunks to the DLQ (Source: `scalability_utils.py`, `StreamForensics.audit_chunk`; `src/main.py`, `train_epoch`; `03_generate_input_data.py`, `execute_etl_stream`).

---

## 2. Technical Architecture

### 2.1. Design Patterns
*   **Self-contained utility pattern:** `scalability_utils.py` is copied into `src/` during job submission to avoid external path dependencies (Source: `01_submit_job.py`).
*   **Dual-metric evaluation:** Regression and classification metrics are computed from final epoch predictions, with AUC only logged when both classes are present (Source: `src/main.py`).

### 2.2. System Integrity and Error Handling
*   **Dead Letter Queue (DLQ):** Chunks that raise processing exceptions during ETL are appended to `inputs/dlq_failed_chunks.csv` (Source: `03_generate_input_data.py`).
*   **Inference Execution:** Worker initialisation uses joblib.load(..., mmap_mode="r") to request memory-mapped reads where supported (Source: `04_local_model_scoring.py`, `init_worker`).
*   **Integrity Auditing:** Retrieval logic records zero-byte warnings and writes `download_manifest.json`, terminating when mandatory artefacts are missing (Source: `02_download_outputs.py`, `perform_integrity_check`, `main`).

---

## 3. Directory Manifest

```text
Graduate_Admissions/
- inputs/                     # Staging for raw data and DLQ output
- outputs/                    # Model artefacts and scoring outputs
- logs/                       # Runtime logs and job state (created at runtime)
- src/                        # Cloud execution context
- src/main.py                 # Entry point: SGD training pipeline
- downloaded_artifacts/       # Retrieved Azure ML artefacts
- scalability_utils.py        # Resource governance utilities
- 01_submit_job.py            # Azure ML job submission
- 02_download_outputs.py      # Artefact retrieval and integrity audit
- 03_generate_input_data.py   # Chunked data preparation and DLQ capture
- 04_local_model_scoring.py   # Parallel batch scoring
- 05_archive_job.py           # Azure job archival
- 06_cleanup_local.py         # Workspace sanitisation
- audit_report.md             # Local audit note
- config.json                 # Local configuration
- config_template.json        # Configuration template
- requirements.txt            # Python dependencies
```

---

## 4. Standard Operating Procedure (SOP)

### 4.1. Prerequisites
Ensure `config.json` is derived from `config_template.json` with valid Azure identifiers.

### 4.2. Execution Pipeline
1.  **Cloud Training**  
    Orchestrates packaging and submits the incremental training job to Azure Compute.
    ```bash
    python 01_submit_job.py
    ```
2.  **Retrieve Artefacts**  
    Download job outputs and run integrity checks after the job completes.
    ```bash
    python 02_download_outputs.py
    ```
3.  **Data Preparation**  
    Stream raw data and normalise for blind inference via chunked ETL.
    ```bash
    python 03_generate_input_data.py
    ```
4.  **Execute Scoring**  
    Run the inference engine using parallel workers; model loading requests memory-mapped reads where supported.
    ```bash
    python 04_local_model_scoring.py
    ```
5.  **Lifecycle Management**  
    Archive the Azure job and sanitise the local workspace.
    ```bash
    python 05_archive_job.py
    python 06_cleanup_local.py
    ```

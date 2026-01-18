# Student Success Classification: Incremental Predictive Pipeline

**Author:** Shane Lee  
**Licence:** MIT  

---

## 1. System Overview

**Operational Capability**  
This module implements a multi-class classification workflow for predicting student success outcomes. Training reads CSV data in chunks and updates the model incrementally; scoring reads CSV data in chunks for inference, avoiding full dataset materialisation (Source: `src/main.py`, `04_local_model_scoring.py`).

**Technical Implementation**  
The training worker uses `SGDClassifier` with `loss='log_loss'` and updates via `partial_fit` on streamed CSV chunks (Source: `src/main.py`). Training memory use is bounded by batch size, model state, class count, and the `StreamForensics` Bloom filter sizing; scoring operates on batch-sized chunks and the loaded model (Source: `src/main.py`, `04_local_model_scoring.py`, `scalability_utils.py`).

*   **Resource Governance:** Batch size resolution can use container cgroup (V1/V2) memory limits when configuration does not set a fixed batch size (Source: `scalability_utils.py`).
*   **Incremental Evaluation:** Weighted metrics are derived from a cumulative confusion matrix with memory proportional to the square of the class count (Source: `src/main.py`).

---

## 2. Technical Architecture

### 2.1. Design Patterns
*   **Self-contained utility pattern:** `01_submit_job.py` packages `scalability_utils.py` into `src` for cloud execution to avoid external path dependencies.
*   **Asynchronous I/O barrier:** ETL writes use a single-thread executor with an explicit write barrier to preserve ordering (Source: `03_generate_input_data.py`).

### 2.2. System Integrity and Error Handling
*   **Dead Letter Queue (DLQ):** Batch processing failures are written to `inputs/dlq_failed_chunks.csv` (Source: `03_generate_input_data.py`).
*   **Memory-mapped model loading:** Worker initialisation loads the model with `joblib.load(..., mmap_mode="r")` to enable memory-mapped reads where supported (Source: `04_local_model_scoring.py`).
*   **Integrity Auditing:** Retrieval logic flags zero-byte files, verifies required artefacts, and records file paths and sizes in `download_manifest.json` (Source: `02_download_outputs.py`).

---

## 3. Directory Manifest

```text
Student_Success/
- inputs/                 # Staging for raw datasets and DLQ logs
- outputs/                # Prediction results and logs
- logs/                   # Operation ledgers and job history
- src/                    # Cloud execution context
- src/main.py             # Entry Point: Incremental multi-class training
- scalability_utils.py    # Master resource governance utility
- 01_submit_job.py        # Azure ML job orchestration
- 02_download_outputs.py  # Artifact retrieval and integrity audit
- 03_generate_input_data.py # Async data streamer and normaliser
- 04_local_model_scoring.py # Parallel batch scoring engine
- 05_archive_job.py       # Azure resource lifecycle management
- 06_cleanup_local.py     # Workspace sanitisation
- config.json             # Local credentials and settings
- config_template.json    # Standardised configuration contract
- requirements.txt        # Module-specific requirement set
```

---

## 4. Standard Operating Procedure (SOP)

### 4.1. Prerequisites
Ensure `config.json` is derived from `config_template.json` and populated with valid Azure identifiers.
```bash
pip install -r requirements.txt
```

### 4.2. Execution Pipeline
1.  **Cloud Training**  
    Dispatch the incremental workload to Azure Compute.
    ```bash
    python 01_submit_job.py
    ```
2.  **Retrieve Artifacts**  
    Download the trained model and audit logs once the job is completed.
    ```bash
    python 02_download_outputs.py
    ```
3.  **Data Preparation**  
    Ingest raw data and normalise for blind inference via asynchronous ETL.
    ```bash
    python 03_generate_input_data.py
    ```
4.  **Execute Scoring**  
    Run the parallel inference engine to generate outcome predictions.
    ```bash
    python 04_local_model_scoring.py
    ```
5.  **Lifecycle Management**  
    Archive the cloud job and sanitise the local environment.
    ```bash
    python 05_archive_job.py
    python 06_cleanup_local.py
    ```

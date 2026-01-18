# Feedback Classification Module: Incremental Text Classification

**Author:** Shane Lee  
**Licence:** MIT  

---

## 1. System Overview

**Operational Capability**  
This module performs incremental training and batch inference for unstructured feedback classification. Training reads CSV input in chunks, applies a HashingVectorizer and PassiveAggressiveClassifier with `partial_fit`, and writes a model artefact plus forensic audit outputs to `outputs/` (Source: `src/main.py`). Artefact retrieval produces a `download_manifest.json` ledger with zero-byte checks and attempts to copy report files into `outputs/` when present (Source: `02_download_outputs.py`). Local scoring streams input in chunks through parallel workers and writes `scoring_results_prediction.csv` (Source: `04_local_model_scoring.py`).

**Technical Implementation**  
The training worker uses a HashingVectorizer with configurable `n_features` (default 1,048,576) and a PassiveAggressiveClassifier updated via `partial_fit` (Source: `src/main.py`). Chunked CSV readers are used for training, ETL preparation, and scoring to bound per-batch memory usage (Source: `src/main.py`, `03_generate_input_data.py`, `04_local_model_scoring.py`).

*   **Memory Profile:** Chunked ingestion bounds per-batch memory. Training retains evaluation buffers from the final epoch for metrics and reporting (Source: `src/main.py`).
*   **Time Complexity:** Training performs a linear pass per epoch over the input stream. Scoring performs a linear pass over the scoring input (Source: `src/main.py`, `04_local_model_scoring.py`).
*   **Resource Governance:** Batch size can be configured or derived from container cgroup limits via `resolve_batch_size`, used in ETL and scoring (Source: `scalability_utils.py`, `03_generate_input_data.py`, `04_local_model_scoring.py`).
*   **Inference Execution:** Parallel scoring uses `ProcessPoolExecutor` with `joblib.load(..., mmap_mode="r")` in worker initialisation to request memory-mapped reads where supported (Source: `04_local_model_scoring.py`).

---

## 2. Technical Architecture

### 2.1. Design Patterns
*   **Self-contained utility pattern:** `scalability_utils.py` is copied into `src/` during job submission to avoid external path dependencies (Source: `01_submit_job.py`).
*   **Hashing feature space:** Hashing-based vectorisation uses a configurable feature space (default 1,048,576 features) (Source: `src/main.py`).

### 2.2. System Integrity and Error Handling
*   **Dead Letter Queue (DLQ):** Chunks that raise processing exceptions during ETL are appended to `inputs/dlq_failed_chunks.csv` (Source: `03_generate_input_data.py`).
*   **Memory mapped inference:** Worker initialisation loads the model with `mmap_mode="r"` to request read-only memory-mapped reads where supported (Source: `04_local_model_scoring.py`).
*   **Integrity Auditing:** Retrieval logic records zero-byte warnings, writes `download_manifest.json`, and attempts report copying from downloaded outputs (Source: `02_download_outputs.py`).

---

## 3. Directory Manifest

```text
Feedback_Classification/
- inputs/                     # Raw feedback streams and DLQ logs
- outputs/                    # Scoring outputs and reports
- logs/                       # Runtime logs and job state (created at runtime)
- src/                        # Cloud execution context
- src/main.py                 # Entry point: incremental NLP training worker
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
2.  **Retrieve Artefacts**  
    Download job outputs and run integrity checks once the job is completed.
    ```bash
    python 02_download_outputs.py
    ```
3.  **Data Preparation**  
    Ingest raw feedback and drop the label column for blind inference.
    ```bash
    python 03_generate_input_data.py
    ```
4.  **Execute Scoring**  
    Run the parallel inference engine to generate classification outputs.
    ```bash
    python 04_local_model_scoring.py
    ```
5.  **Lifecycle Management**  
    Archive the cloud job and sanitise the local environment.
    ```bash
    python 05_archive_job.py
    python 06_cleanup_local.py
    ```

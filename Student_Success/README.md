# Student Success Classification: Incremental Predictive Pipeline

**Author:** Shane Lee  
**Licence:** MIT  
**Version:** 1.0.0  

---

## 1. Executive Summary

**Operational Capability:**  
This module implements a multi-class classification system for predicting student success outcomes. The system utilises an out-of-core architecture to decouple memory requirements from data volume, enabling the processing of datasets exceeding $10^7$ rows on standard compute hardware. This facilitates stable infrastructure expenditure as data scales, providing a solution for institutional analytics while maintaining a verifiable chain of custody for all predictive artifacts.

**Technical Implementation:**  
The system implements an incremental learning pipeline utilising the `SGDClassifier` with a log_loss objective. It is engineered for strictly $O(1)$ space complexity during the training and scoring phases.  
*   **Space Complexity:** Strictly $O(1)$ Resident Set Size (RSS) via chunked streaming generators and online fitment.
*   **Time Complexity:** $O(N)$ linear pass for ingestion, training, and inference.
*   **Resource Governance:** Dynamic batch size resolution through inspection of container cgroup (V1/V2) memory limits.
*   **Incremental Evaluation:** Derivation of weighted metrics (accuracy, precision, recall, F1) from a cumulative confusion matrix to ensure constant memory overhead during validation.

---

## 2. Technical Architecture

### 2.1. Verified Design Patterns
*   **Self-Contained Utility Pattern:** Implements a local injection of `scalability_utils.py` into the deployment context to ensure environment parity during cloud dispatch without external path dependencies.
*   **Asynchronous I/O Barrier:** ETL streams utilise threaded serialisation to mitigate storage latency during high-throughput ingestion.

### 2.2. System Integrity and Error Handling
*   **Dead Letter Queue (DLQ):** Schema violations are isolated to `inputs/dlq_failed_chunks.csv`, ensuring stream continuity and enabling analysis of malformed records.
*   **Zero-Copy Inference:** Utilises kernel-level memory mapping via `joblib` to share model segments across parallel worker processes, minimising physical RAM consumption.
*   **Integrity Auditing:** Retrieval logic executes mandatory zero-byte audits on cloud-downloaded artifacts to detect transfer corruption before local promotion.

---

## 3. Directory Manifest

```text
Student_Success/
├── inputs/                 # Staging for raw datasets and DLQ logs
├── outputs/                # Prediction results and logs
├── logs/                   # Operation ledgers and job history
├── src/                    # Cloud execution context
│   └── main.py             # Entry Point: Incremental multi-class training
├── scalability_utils.py    # Master resource governance utility
├── 01_submit_job.py        # Azure ML job orchestration
├── 02_download_outputs.py  # Artifact retrieval and integrity audit
├── 03_generate_input_data.py # Async data streamer and normaliser
├── 04_local_model_scoring.py # Parallel batch scoring engine
├── 05_archive_job.py       # Azure resource lifecycle management
├── 06_cleanup_local.py     # Workspace sanitisation
├── config.json             # Local credentials and settings
├── config_template.json    # Standardised configuration contract
└── requirements.txt        # Module-specific requirement set
```

---

## 4. Standard Operating Procedure (SOP)

### 4.1. Prerequisites
Ensure `config.json` is derived from `config_template.json` and populated with valid Azure identifiers.
```bash
pip install -r requirements.txt
```

### 4.2. Execution Pipeline
1.  **Cloud Training:** Dispatch the incremental workload to Azure Compute.
    ```bash
    python 01_submit_job.py
    ```
2.  **Retrieve Artifacts:** Download the trained model and audit logs once the job is completed.
    ```bash
    python 02_download_outputs.py
    ```
3.  **Data Preparation:** Ingest raw data and normalise for blind inference via asynchronous ETL.
    ```bash
    python 03_generate_input_data.py
    ```
4.  **Execute Scoring:** Run the parallel inference engine to generate outcome predictions.
    ```bash
    python 04_local_model_scoring.py
    ```
5.  **Lifecycle Management:** Archive the cloud job and sanitise the local environment.
    ```bash
    python 05_archive_job.py
    python 06_cleanup_local.py
    ```
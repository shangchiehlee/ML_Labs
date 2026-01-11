# Feedback Classification Module: Incremental Predictive Engine

**Author:** Shane Lee  
**Licence:** MIT  
**Version:** 1.0.0  

---

## 1. Executive Summary

**Operational Capability:**  
This module implements a system for classifying unstructured feedback data. By employing an out-of-core architectural pattern, the system decouples data volume from memory requirements, facilitating the analysis of multi-million row datasets on standard compute hardware. This implementation ensures predictable resource costs by eliminating the requirement for high-memory infrastructure whilst maintaining a verifiable chain of custody for all predictive outputs.

**Technical Implementation:**  
The system implements a binary text classification pipeline engineered for strictly $O(1)$ space complexity. Feature projection is executed via a stateless HashingVectorizer employing the MurmurHash3 algorithm, which removes the requirement for a memory-resident vocabulary and ensures a constant memory footprint regardless of linguistic diversity. The model utilises a PassiveAggressiveClassifier updated through incremental fitment.  
*   **Space Complexity:** Strictly $O(1)$ Resident Set Size (RSS) via chunked streaming and stateless vectorisation.
*   **Time Complexity:** $O(N)$ linear pass for training and inference phases.
*   **Resource Governance:** Dynamic batch resolution based on container cgroup (V1/V2) memory limits.
*   **Inference Performance:** Parallel batch processing utilising kernel-level memory mapping to share model segments across worker processes.

---

## 2. Technical Architecture

### 2.1. Core Design Patterns
*   **Self-Contained Utility Pattern:** Implements a local injection of `scalability_utils.py` into the deployment context to ensure atomic cloud execution without external path dependencies.
*   **Stateless Feature Engineering:** Utilises deterministic hashing to project text into high-dimensional space ($2^{20}$ dimensions) with zero RAM growth relative to vocabulary size.

### 2.2. System Integrity and Error Handling
*   **Dead Letter Queue (DLQ):** Malformed records in ETL streams are isolated to `inputs/dlq_failed_chunks.csv` for analysis, ensuring stream continuity.
*   **Zero-Copy Inference:** Utilises kernel-level memory mapping via `joblib` to share model segments across parallel workers, minimising physical memory overhead.
*   **Integrity Auditing:** Retrieval logic executes mandatory zero-byte checks on all cloud-downloaded artifacts to detect transfer corruption before local promotion.

---

## 3. Directory Manifest

```text
Feedback_Classification/
├── inputs/                 # Raw feedback streams and DLQ logs
├── outputs/                # Sentiment drivers and reports
├── logs/                   # Operation ledgers and job history
├── src/                    # Cloud execution context
│   └── main.py             # Entry Point: Incremental NLP pipeline
├── downloaded_artifacts/   # Trained models retrieved from Azure
├── scalability_utils.py    # Master resource governance utility
├── 01_submit_job.py        # Azure ML job orchestrator
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
2.  **Retrieve Artifacts:** Download the model and audits once the job is completed.
    ```bash
    python 02_download_outputs.py
    ```
3.  **Data Preparation:** Ingest raw feedback and normalise for blind inference.
    ```bash
    python 03_generate_input_data.py
    ```
4.  **Execute Scoring:** Run the parallel inference engine to generate classification outputs.
    ```bash
    python 04_local_model_scoring.py
    ```
5.  **Lifecycle Management:** Archive the cloud job and sanitise the local environment.
    ```bash
    python 05_archive_job.py
    python 06_cleanup_local.py
    ```
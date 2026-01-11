# Graduate Admissions Assessment Module: Incremental Predictive Pipeline

**Author:** Shane Lee  
**Licence:** MIT  
**Version:** 1.0.0  

---

## 1. Executive Summary

**Operational Capability:**  
This module implements an incremental learning architecture for estimating graduate admission probabilities. The decoupling of dataset volume from memory requirements enables the processing of datasets exceeding $10^7$ rows on standard compute instances. This implementation maintains stable infrastructure costs for large-scale educational analytics by transitioning from in-memory processing to a streaming paradigm.

**Technical Implementation:**  
The system implements a stochastic gradient descent regression pipeline engineered for strictly $O(1)$ space complexity. It adheres to an incremental learning paradigm, ensuring that the Resident Set Size (RSS) remains constant regardless of input volume.  
*   **Incremental Learning:** Utilises `SGDRegressor.partial_fit()` with an adaptive learning rate to update model weights on streaming batches.
*   **Memory Decoupling:** Enforced via chunked reader generators and online fitment.
*   **Time Complexity:** $O(N)$ linear pass for training and inference phases.
*   **Resource Governance:** Dynamically resolves optimal batch sizes through inspection of container cgroup (V1/V2) memory limits.
*   **Probabilistic Auditing:** Implements deduplication via Bloom filters ($O(m)$ space) and isolated schema violation routing via a dead letter queue.

---

## 2. Technical Architecture

### 2.1. Verified Design Patterns
*   **Self-Contained Utility Pattern:** Utilises a local injection of `scalability_utils.py` to ensure atomic cloud deployment without external path dependencies. The training worker performs dual-task evaluation: deriving regression metrics (R2, MSE, MAE) and classification metrics (accuracy, F1, AUC) from the continuous stream to monitor convergence.

### 2.2. System Integrity and Error Handling
*   **Dead Letter Queue (DLQ):** Malformed records in ETL streams are isolated to `inputs/dlq_failed_chunks.csv`, ensuring stream continuity and enabling analysis.
*   **Inference Performance:** Scoring workers utilise `joblib` memory mapping to share physical memory pages across parallel processes, reducing physical RAM overhead.
*   **Integrity Auditing:** Retrieval logic executes mandatory zero-byte audits on cloud-downloaded artifacts to detect transfer corruption before local promotion.

---

## 3. Directory Manifest

```text
Graduate_Admissions/
├── inputs/                 # Staging for raw data and DLQ output
├── outputs/                # Prediction results and logs
├── logs/                   # Operation ledgers and job history
├── src/                    # Cloud execution context
│   └── main.py             # Entry Point: SGD training pipeline
├── downloaded_artifacts/   # Models retrieved from Azure
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

**Prerequisite:** Ensure `config.json` is derived from `config_template.json` with valid Azure identifiers.

### 4.1. Cloud Training Phase
1.  **Dispatch Workload:** Orchestrates packaging and submits the incremental training job to Azure Compute.
    ```bash
    python 01_submit_job.py
    ```
2.  **Retrieve Artifacts:** Download the trained model and audits once the job is completed.
    ```bash
    python 02_download_outputs.py
    ```

### 4.2. Local Inference Phase
1.  **Prepare Blind Data:** Stream raw data and normalise for blind inference via asynchronous ETL.
    ```bash
    python 03_generate_input_data.py
    ```
2.  **Execute Scoring:** Run the inference engine using parallel memory-mapped workers to generate predictions.
    ```bash
    python 04_local_model_scoring.py
    ```

### 4.3. Lifecycle Management
1.  **Archive Job:** Soft-deletes the Azure job to maintain workspace hygiene.
    ```bash
    python 05_archive_job.py
    ```
2.  **Sanitise Workspace:** Recursively removes temporary data and logs.
    ```bash
    python 06_cleanup_local.py
    ```

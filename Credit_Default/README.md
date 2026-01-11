# Credit Default Prediction Module: Incremental Risk Engine

**Author:** Shane Lee  
**Licence:** MIT  
**Version:** 1.0.0  

---

## 1. Executive Summary

**Operational Capability:**  
This module implements a scalable risk scoring engine for predicting credit default probabilities. By utilising an out-of-core architectural pattern, the system decouples data volume from memory requirements, enabling the processing of multi-million row datasets on standard compute instances. This approach ensures infrastructure cost stability as data volumes grow, providing a verifiable and auditable mechanism for large-scale financial analytics.

**Technical Implementation:**  
The module executes a streaming ensemble learning pipeline comprising seven independent Stochastic Gradient Descent (SGD) classifiers, aggregated via soft voting. To capture non-linear decision boundaries within a linear streaming framework, it utilises Nystroem kernel approximation to project features into a 1200-dimensional space with $O(n)$ complexity.
*   **Space Complexity:** Strictly $O(1)$ Resident Set Size (RSS) achieved via chunked generators (`pd.read_csv(chunksize=...)`) and `partial_fit` incremental learning.
*   **Time Complexity:** $O(N)$ linear pass for both training and inference phases.
*   **Resource Governance:** Dynamic batch size resolution based on active Linux cgroup (V1/V2) memory limit inspection.
*   **Inference:** Parallel batch processing utilising kernel-level memory mapping (mmap) to share model segments across worker processes.

---

## 2. Technical Architecture

### 2.1. Verified Design Patterns
*   **Self-Contained Utility Pattern:** The module injects a local instance of `scalability_utils.py` into the deployment context, ensuring atomic cloud execution without external path dependencies on the root `Utils/` directory.
*   **Asynchronous I/O Barrier:** ETL streams utilise threaded disk serialisation to prevent storage latency from becoming a bottleneck during ingestion.

### 2.2. System Integrity and Error Handling
*   **Dead Letter Queue (DLQ):** Schema violations and malformed records are isolated to `inputs/dlq_failed_chunks.csv`, ensuring stream continuity while preserving data for forensic analysis.
*   **Probabilistic Auditing:** Utilises Bloom Filters ($O(m)$ space) for efficient membership testing and deduplication of streaming data.
*   **Integrity Verification:** The retrieval logic enforces mandatory zero-byte checks on all cloud-downloaded artifacts to detect transfer corruption before local promotion.

---

## 3. Directory Manifest

```text
Credit_Default/
├── inputs/                     # Staging for raw data and DLQ logs (GitIgnored)
├── outputs/                    # Inference results and metrics (GitIgnored)
├── logs/                       # Operation ledgers and forensic logs (GitIgnored)
├── src/                        # Remote execution context
│   └── main.py                 # Entry Point: Incremental ensemble training script
├── scalability_utils.py        # Master resource governance utility (Local Instance)
├── 01_submit_job.py            # Azure ML job orchestration utility
├── 02_download_outputs.py      # Artifact retrieval and forensic audit utility
├── 03_generate_input_data.py   # Asynchronous streaming ETL and validation utility
├── 04_local_model_scoring.py   # Parallel batch inference engine
├── 05_archive_job.py           # Lifecycle management for Azure resources
├── 06_cleanup_local.py         # Workspace sanitisation utility
├── config.json                 # Runtime credentials and environment settings (GitIgnored)
├── config_template.json        # Public schema contract for configuration
└── requirements.txt            # Module-specific dependency manifest
```

---

## 4. Standard Operating Procedure (SOP)

### 4.1. Prerequisites
Ensure `config.json` is derived from `config_template.json` and populated with valid Azure identifiers.
```bash
pip install -r requirements.txt
```

### 4.2. Execution Pipeline

1.  **Dispatch Training Workload:**
    Packages the source context and submits the incremental ensemble job to Azure Compute.
    ```bash
    python 01_submit_job.py
    ```

2.  **Retrieve and Audit Artifacts:**
    Downloads trained models and logs, performing immediate integrity checks.
    ```bash
    python 02_download_outputs.py
    ```

3.  **Stream Ingestion (ETL):**
    Streams raw data, strips labels for blind inference, and validates schema conformance.
    ```bash
    python 03_generate_input_data.py
    ```

4.  **Execute Local Inference:**
    Runs the scoring engine using parallel memory-mapped workers to generate risk probabilities.
    ```bash
    python 04_local_model_scoring.py
    ```

5.  **Lifecycle Management:**
    Archives the Azure job and sanitises the local workspace (logs, temp inputs).
    ```bash
    python 05_archive_job.py
    python 06_cleanup_local.py
    ```

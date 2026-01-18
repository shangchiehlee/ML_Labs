# Credit Default Prediction Module: Incremental Risk Engine

**Author:** Shane Lee  
**Licence:** MIT  

---

## 1. System Overview

**Operational Capability**  
This module implements a risk scoring engine for estimating credit default probabilities. Training and scoring scripts stream CSV data in chunks rather than loading full datasets into memory. Training performs a class weight pre-scan and accumulates final-epoch test predictions in memory for metric calculation.

**Technical Implementation**  
The module executes a streaming ensemble learning pipeline comprising a configurable number of Stochastic Gradient Descent (SGD) classifiers, aggregated via soft voting. It uses Nystroem kernel approximation to project features into a `nystroem_components`-dimensional space, with a default of 1200 defined in `training_hyperparameters` when not overridden at submission.

*   **Memory Profile:** Training and inference stream data in chunks using `pd.read_csv(chunksize=...)`. Training accumulates final-epoch test outputs in memory for metric calculation (Source: `src/main.py`).
*   **Training Passes:** A pre-scan derives class weights, followed by `max_iter` epochs over the dataset (Source: `src/main.py`).
*   **Resource Governance:** ETL and local scoring resolve batch size via `resolve_batch_size`, which can derive a batch size from cgroup V1 or V2 memory limits when `batch_size` is unset (Source: `scalability_utils.py`, `03_generate_input_data.py`, `04_local_model_scoring.py`).
*   **Inference:** Worker processes load the model artefact with `joblib.load(..., mmap_mode="r")` and score chunks in parallel (Source: `04_local_model_scoring.py`).

---

## 2. Technical Architecture

### 2.1. Design Patterns
*   **Self-contained utility pattern:** The module copies `scalability_utils.py` into `src` during job submission to avoid dependencies on the root `Utils` directory in the job execution environment (Source: `01_submit_job.py`).
*   **Serialised write barrier:** ETL streams use `ThreadPoolExecutor` with a single worker and wait on the prior future to preserve write ordering (Source: `03_generate_input_data.py`).

### 2.2. System Integrity and Error Handling
*   **Dead Letter Queue (DLQ):** Chunks that fail schema validation are written to `inputs/dlq_failed_chunks.csv`, and subsequent chunks continue processing.
*   **Probabilistic Auditing:** The system utilises Bloom filters ($O(m)$ space) for membership testing and deduplication of streaming data (Source: `scalability_utils.py`).
*   **Integrity Verification:** The retrieval logic records zero-byte warnings and writes a `download_manifest.json` ledger, terminating if mandatory artefacts are missing (Source: `02_download_outputs.py`).

---

## 3. Directory Manifest

```text
Credit_Default/
- inputs/                    # Staging for raw data and DLQ logs (GitIgnored)
- outputs/                   # Inference results and metrics (GitIgnored)
- logs/                      # Operation ledgers and forensic logs (GitIgnored)
- src/                       # Remote execution context
- src/main.py                # Entry point for incremental ensemble training
- scalability_utils.py       # Master resource governance utility (Local Instance)
- 01_submit_job.py           # Azure ML job orchestration utility
- 02_download_outputs.py     # Artifact retrieval and forensic audit utility
- 03_generate_input_data.py  # Streaming ETL and validation utility
- 04_local_model_scoring.py  # Parallel batch inference engine
- 05_archive_job.py          # Lifecycle management for Azure resources
- 06_cleanup_local.py        # Workspace sanitisation utility
- config.json                # Runtime configuration (GitIgnored)
- config_template.json       # Configuration template
- requirements.txt           # Module-specific dependency manifest
```

---

## 4. Standard Operating Procedure (SOP)

### 4.1. Prerequisites
Ensure `config.json` is derived from `config_template.json` and populated with valid Azure identifiers.
```bash
pip install -r requirements.txt
```

### 4.2. Execution Pipeline

1.  **Dispatch Training Workload**  
    Packages the source context and submits the incremental ensemble job to Azure Compute.
    ```bash
    python 01_submit_job.py
    ```

2.  **Retrieve and Audit Artifacts**  
    Downloads trained models and logs, performing immediate integrity checks.
    ```bash
    python 02_download_outputs.py
    ```

3.  **Stream Ingestion (ETL)**  
    Streams raw data, strips labels for blind inference, and validates schema conformance.
    ```bash
    python 03_generate_input_data.py
    ```

4.  **Execute Local Inference**  
    Runs the scoring engine using parallel memory-mapped workers to generate risk probabilities.
    ```bash
    python 04_local_model_scoring.py
    ```

5.  **Lifecycle Management**  
    Archives the Azure job and sanitises the local workspace (logs, temp inputs).
    ```bash
    python 05_archive_job.py
    python 06_cleanup_local.py
    ```

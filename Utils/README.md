# ML_Labs Foundation: Infrastructure and Diagnostic Layer

**Author:** Shane Lee  
**Licence:** MIT  

---

## 1. System Overview

**Operational Capability**  
The Utils module provides shared utilities and diagnostics used by ML_Labs scripts. It includes configuration loading, logging setup, memory limit inspection, batch size resolution, and diagnostics for Azure ML connectivity and scikit-learn estimator availability.

**Technical Implementation**  
Core utilities live in `scalability_utils.py` and include configuration loading (`load_config`), logging setup (`configure_logging`), schema validation (`validate_dataframe_schema`), memory limit inspection (`get_memory_limit`), batch size resolution (`resolve_batch_size`), and streaming forensics helpers (`BloomFilter`, `StreamForensics`). Diagnostic scripts are `connect_to_workspace.py` and `audit_model_inventory.py`.

*   **Resource Governance:** Memory limit detection and batch size resolution helpers (`get_memory_limit`, `calculate_optimal_batch_size`, `resolve_batch_size`) (Source: `scalability_utils.py`).
*   **Probabilistic Membership Testing:** Bloom filter membership checks used by `StreamForensics.audit_chunk` for repeated index filtering (Source: `scalability_utils.py`).
*   **Logging:** JSON or text logging output via `JSONFormatter` and `configure_logging` (Source: `scalability_utils.py`).
*   **Diagnostics:** Azure ML connectivity check via `workspaces.get` with latency reporting (Source: `connect_to_workspace.py`), and estimator inventory via `sklearn.utils.all_estimators` (Source: `audit_model_inventory.py`).

---

## 2. Technical Architecture

### 2.1. Design Patterns
The module exposes shared utilities in `scalability_utils.py` and diagnostic scripts that import those utilities for connectivity and estimator checks.

### 2.2. System Integrity and Error Handling
*   **Identity Negotiation:** `connect_to_workspace.py` checks for required configuration keys, attempts DefaultAzureCredential authentication, and exits non-zero on failure.
*   **Recursive Sanitisation:** `06_cleanup_local.py` uses permission-aware deletion to retry removals on read-only files.
*   **Schema Validation:** `load_config` logs schema version mismatches and `validate_dataframe_schema` raises on missing required columns.

---

## 3. Directory Manifest

```text
Utils/
- inputs/                  # Runtime staging
- outputs/                 # Runtime outputs
- logs/                    # Runtime logs (created by configure_logging)
- scalability_utils.py     # Shared utilities for configuration and logging
- connect_to_workspace.py  # Azure ML connectivity diagnostics
- audit_model_inventory.py # Scikit-learn estimator inventory
- 06_cleanup_local.py      # Workspace sanitisation
- audit_report.md          # Local audit note
- config.json              # Local configuration
- config_template.json     # Configuration template
- requirements.txt         # Python dependencies
```

---

## 4. Standard Operating Procedure (SOP)

### 4.1. Prerequisites
Ensure the root virtual environment is active and install the foundational requirements:
```bash
pip install -r requirements.txt
```

### 4.2. Execution Pipeline
1.  **Connectivity Audit**  
    Verify Azure Machine Learning connectivity and connection latency.
    ```bash
    python connect_to_workspace.py
    ```
2.  **Runtime Dependency Inspection**  
    Audit the local environment for available classification estimators.
    ```bash
    python audit_model_inventory.py
    ```
3.  **Environment Sanitisation**  
    Remove local runtime directories and logs.
    ```bash
    python 06_cleanup_local.py
    ```

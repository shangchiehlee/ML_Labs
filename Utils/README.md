# ML_Labs Foundation: Infrastructure and Diagnostic Layer

**Author:** Shane Lee  
**Licence:** MIT  
**Version:** 1.0.0  

---

## 1. Executive Summary

**Operational Capability:**  
The Utils module implements the foundational infrastructure for the ML_Labs monorepo, establishing standardised mechanisms for resource governance and environment synchronisation. By implementing automated diagnostic and deployment primitives, the module reduces operational failure rates and enables compute cost containment. The architectural pattern utilised ensures that individual machine learning modules operate within strictly defined resource bounds, allowing large-scale data processing on standard hardware.

**Technical Implementation:**  
This module is the authoritative source for the monorepo's shared primitives, engineered for strictly $O(1)$ space complexity and container-aware execution.  
*   **Resource Governance:** Logic for inspecting Linux cgroup (V1/V2) memory limits to identify execution bounds and prevent out-of-memory termination.
*   **Probabilistic Membership Testing:** Implementation of Bloom filters ($O(m)$ space) for membership verification across streaming datasets.
*   **Environment Parity:** Provides the master `scalability_utils.py` library, which is injected into cloud execution contexts during orchestration to ensure parity between local and remote runtimes.
*   **Diagnostics:** Azure Machine Learning handshaking with latency measurements and scikit-learn registry auditing.

---

## 2. Technical Architecture

### 2.1. Authoritative Design Patterns
The module implements the **Self-Contained Utility Pattern**, acting as the repository for infrastructure logic that is mirrored across modules to eliminate cross-module relative path dependencies. It provides a logging factory for serialising machine-parseable JSON or text output.

### 2.2. System Integrity and Error Handling
*   **Identity Negotiation:** `connect_to_workspace.py` performs validation of the `DefaultAzureCredential` resolution chain, providing diagnostic output for identity and access management failures.
*   **Recursive Sanitisation:** `06_cleanup_local.py` utilises permission-aware deletion routines to handle restricted file attributes and file locking.
*   **Schema Validation:** Configuration utilities enforce versioned schema validation to ensure environment configuration parity.

---

## 3. Directory Manifest

```text
Utils/
├── inputs/                 # Diagnostic staging
├── outputs/                # Audit ledgers and logs
├── logs/                   # Operation history
├── scalability_utils.py    # Master resource governance utility
├── connect_to_workspace.py # Azure ML connectivity handshake
├── audit_model_inventory.py # Scikit-learn registry introspection
├── 06_cleanup_local.py     # Workspace sanitisation
├── audit_report.md         # Quality assurance sign-off
├── config.json             # Local credentials and settings
├── config_template.json    # Foundation schema contract
└── requirements.txt        # Infrastructure dependency set
```

---

## 4. Standard Operating Procedure (SOP)

### 4.1. Setup and Environment Initialisation
Ensure the root virtual environment is active and install the foundational requirements:
```bash
pip install -r requirements.txt
```

### 4.2. Pre-Flight Connectivity Audit
Verify the Azure Machine Learning identity handshake and connection latency:
```bash
python connect_to_workspace.py
```

### 4.3. Runtime Dependency Inspection
Audit the local environment for available classification estimators:
```bash
python audit_model_inventory.py
```

### 4.4. Environment Sanitisation
Execute a hard reset of the local environment to purge transient artifacts and logs:
```bash
python 06_cleanup_local.py
```
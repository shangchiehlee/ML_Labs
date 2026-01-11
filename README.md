# ML_Labs: Incremental Machine Learning Monorepo

**Author:** Shane Lee  
**Licence:** MIT  
**Version:** 1.0.0  

---

## 1. Executive Summary

ML_Labs is a monorepo containing independent machine learning modules engineered for large-scale data processing. The system architecture enforces an out-of-core compute model designed to decouple memory consumption from dataset volume, enabling training and inference on datasets exceeding physical memory ($N > 10^7$ rows). By adhering to incremental learning paradigms and kernel-level resource optimisation, the system maintains infrastructure cost stability and predictable throughput for massive data ingestion.

---

## 2. Global Architecture and System Integration

### 2.1. Shared Infrastructure: The Self-Contained Utility Pattern
The repository implements a decoupled dependency model to ensure atomic cloud deployment and environment parity:
*   **Infrastructure Source:** The `Utils/` directory serves as the repository for diagnostic scripts and the master `scalability_utils.py` library.
*   **Module-Level Mirroring:** Each module maintains a local copy of `scalability_utils.py` to facilitate isolated development and testing without cross-module relative path dependencies.
*   **Orchestration Injection:** During the build phase, the orchestration logic (`01_submit_job.py`) packages the utility into the `src/` distribution. This ensures that remote Azure Machine Learning environments possess identical primitives to the local runtime, eliminating import errors in containerised contexts.

### 2.2. Shared Operational Primitives
All modules leverage a centralised resource governance layer defined in `scalability_utils.py`:
*   **Container Awareness:** Memory limit detection prioritises Linux cgroup (V1/V2) inspection over host physical memory to prevent out-of-memory termination.
*   **Zero-Copy Inference:** The system utilises kernel-level memory mapping via `joblib` to share model segments across parallel worker processes, minimising the resident set size.
*   **Data Stream Resilience:** Implementation of Dead Letter Queues (DLQ) for all streaming ETL pipelines to isolate records violating schema constraints without halting execution.

---

## 3. Inventory of Capabilities

| Module | Domain | Executive Summary | Key Mechanisms |
| :--- | :--- | :--- | :--- |
| **[Credit_Default](./Credit_Default)** | Finance | Predicts default probabilities using out-of-core ensemble learning. | Seven SGD classifiers: Nystroem kernel approximation ($O(n)$ space). |
| **[Feedback_Classification](./Feedback_Classification)** | NLP | Classifies unstructured feedback streams utilising stateless feature extraction for $O(1)$ memory. | HashingVectorizer ($2^{20}$ dimensions): PassiveAggressive learning. |
| **[Graduate_Admissions](./Graduate_Admissions)** | Education | Estimates admission probabilities using an incremental regression pipeline. | SGDRegressor: adaptive learning rates: incremental fitment. |
| **[Student_Success](./Student_Success)** | Education | Identifies retention outcomes utilising multi-class prediction and memory-constant matrix auditing. | Multi-class SGD: cumulative confusion matrix: $O(1)$ space evaluation. |
| **[Utils](./Utils)** | MLOps | Executes pre-flight validation: Azure connectivity resolution: and environment resource introspection. | Cgroup detection: Azure ML connectivity diagnostics: algorithm registry auditing. |

---

## 4. Unified Setup Guide

### 4.1. Master Environment Initialisation
The monorepo operates within a single virtual environment anchored at the root.

```bash
# 1. Create virtual environment at root
python -m venv .venv

# 2. Activate (Windows)
.venv\Scripts\activate
# 2. Activate (Linux/MacOS)
source .venv/bin/activate

# 3. Install foundational dependencies
pip install -r Utils/requirements.txt
```

### 4.2. Configuration Protocol
Each module requires a local `config.json` derived from its respective `config_template.json`.

1.  **Navigate** to the target module directory.
2.  **Duplicate** `config_template.json` to `config.json`.
3.  **Populate** identifiers: `subscription_id`: `resource_group`: and `workspace_name`.
4.  **Authenticate** via Azure CLI: `az login`.

---

## 5. Contribution and Governance

### 5.1. Technical Standards
*   **Type Hinting:** Mandatory for all function signatures to enforce contract clarity and facilitate static analysis.
*   **PEP 8 Compliance:** Adherence to Python styling conventions is required.
*   **Documentation:** Docstrings must describe the algorithmic mechanism: resource complexity (time and space): and operational side effects.
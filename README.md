# ML_Labs Monorepo

**Author:** Shane Lee  
**Licence:** MIT  

---

## 1. Repository Overview

ML_Labs is a monorepo containing independent modules:

*   Credit_Default
*   Feedback_Classification
*   Graduate_Admissions
*   Student_Success
*   Utils

Each module has its own directory with scripts, configuration files, and a module README.

---

## 2. Module Model and Lifecycle

The repository follows a module isolation model:

*   Modules are audited and frozen independently.
*   Modules are not refactored across boundaries unless a concrete defect requires it.
*   Shared behaviour is implemented via copied utilities rather than runtime imports across modules.
*   Modules are intended to remain deployable and auditable in isolation.
*   The module lifecycle is: audit and refactor, end-to-end execution validation, behavioural necessity classification, formal freeze.
*   After freeze, a module is modified only when a concrete defect is identified.

---

## 3. Azure ML Packaging Boundary and Utility Injection

Azure ML jobs are submitted with the module `src` directory as the code root (`code = "./src"`). As a result, only files inside the module `src` directory are included in the remote execution context.

Each module includes a `scalability_utils.py` at the module root. During job submission, `01_submit_job.py` copies `scalability_utils.py` into the module `src` directory to satisfy the packaging boundary. The copied file is transient and removed by `06_cleanup_local.py`.

---

## 4. Module Index

| Module | Summary |
| :--- | :--- |
| **[Credit_Default](./Credit_Default)** | Training entry point (`src/main.py`) and local scoring script (`04_local_model_scoring.py`). |
| **[Feedback_Classification](./Feedback_Classification)** | Training entry point (`src/main.py`) and local scoring script (`04_local_model_scoring.py`). |
| **[Graduate_Admissions](./Graduate_Admissions)** | Training entry point (`src/main.py`) and local scoring script (`04_local_model_scoring.py`). |
| **[Student_Success](./Student_Success)** | Training entry point (`src/main.py`) and local scoring script (`04_local_model_scoring.py`). |
| **[Utils](./Utils)** | Diagnostics and shared utilities (`scalability_utils.py`, `connect_to_workspace.py`, `audit_model_inventory.py`). |

---

## 5. Configuration and Dependencies

Each module includes `config_template.json`. Module scripts load configuration via `scalability_utils.load_config`, which resolves `config.json` relative to the module directory.

Each module includes a `requirements.txt` in its directory. The Utils diagnostics use `Utils/requirements.txt`.

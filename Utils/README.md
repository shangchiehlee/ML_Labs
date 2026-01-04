# ML_Labs: Diagnostic & Validation Utilities

**System Owner:** Shane Lee
**Licence:** MIT

## Executive Summary

**Target Audience:** Senior Executives & Project Sponsors

This module functions as a pre-flight validation suite for the ML_Labs environment. It executes connectivity diagnostics, authentication verification, and library introspection to ensure environment parity between local development and cloud execution contexts.

By validating infrastructure dependencies before workload submission, this system mitigates the risk of runtime failures in provisioned cloud environments.

**Operational Capabilities:**
*   **Environment Parity:** Verifies that the local runtime possesses the specific mathematical libraries and versions required for model deployment.
*   **Cost Avoidance:** Prevents the provisioning of billable cloud compute resources if authentication or network paths are misconfigured.
*   **Infrastructure Agnosticism:** Operates consistently across local workstations, Docker containers, and Azure Compute Instances without code modification.

## Technical Architecture

**Target Audience:** Lead Engineers & Architects

The system employs a modular architecture separating configuration, shared infrastructure primitives, and executable diagnostics.

### 1. Identity & Access Management (IAM)
The system abstracts authentication logic using the `azure.identity.DefaultAzureCredential` class within `connect_to_workspace.py`. This enforces a secure hierarchy of authentication methods:
1.  **Environment Variables:** Checks for service principal credentials.
2.  **Managed Identity:** Utilises assigned identities when running within Azure Compute.
3.  **Azure CLI:** Falls back to the local developer's cached `az login` token.

### 2. Resource Governance (Container Awareness)
The `scalability_utils.py` library implements a hierarchical memory detection strategy to support containerised execution:
1.  **Cgroup Inspection:** The system first attempts to read `/sys/fs/cgroup/memory/memory.limit_in_bytes`. If a valid limit (defined as $< 10^{15}$ bytes) is detected, this value is enforced. This ensures the process respects Docker/Kubernetes limits rather than the host's total physical memory.
2.  **Host Fallback:** If Cgroups are inaccessible or report unlimited memory, the system defaults to `psutil.virtual_memory().total`.

### 3. Artifact Management & Locking Prevention
The `06_cleanup_local.py` script implements a defensive deletion routine:
*   **Lock Avoidance:** It initialises a transient, console-only logging configuration. This deliberately bypasses the standard file-logging factory to prevent OS-level `PermissionError` conflicts (common on Windows) that occur when a process holds a file handle within a directory it attempts to delete.
*   **Bytecode Sanitisation:** The script executes a recursive `os.walk` ($O(N)$ complexity) to identify and remove `__pycache__` directories, ensuring a clean state for subsequent executions.

## System Integrity

### 1. Logging Architecture
The `scalability_utils.setup_logger` function implements a dual-handler pattern:
*   **StreamHandler:** Directs logs to `sys.stdout` for real-time console monitoring.
*   **FileHandler:** Persists logs to the `logs/` directory for audit trails.
*   **Handler Management:** The factory explicitly clears existing handlers (`logger.handlers.clear()`) prior to initialisation. This prevents duplicate log entries when scripts are executed repeatedly within the same Python runtime.

### 2. Configuration Validation
The system enforces strict schema validation via `scalability_utils.load_configuration`:
*   **Existence Check:** Verifies the presence of `config.json`.
*   **Syntax Validation:** Catches `json.JSONDecodeError` to prevent partial loading of malformed configurations.
*   **Failure State:** Triggers an immediate `sys.exit(1)` upon validation failure, preventing undefined behaviour.

## Directory Manifest

```text
Utils/
│
├── audit_model_inventory.py   # Introspection: Enumerates scikit-learn classifiers via registry. Validated against scikit-learn v1.5.1 (43 algorithms available as of 2026-01-03).
├── connect_to_workspace.py    # Diagnostics: Validates Azure MLClient handshake and latency.
├── 06_cleanup_local.py        # Maintenance: Sanitises logs/ and bytecode artifacts.
├── scalability_utils.py       # Library: Shared logic for logging, config, and memory checks.
├── config.json                # Configuration: Infrastructure identifiers (GitIgnored).
└── logs/                      # Artifacts: Runtime operation logs.
```

## Standard Operating Procedure (SOP)

### Phase 1: Configuration
1.  **Credential Setup:** Create a `config.json` file in the root directory. Define the `subscription_id`, `resource_group`, and `workspace_name`.
2.  **Authentication:** Execute `az login` in the terminal to cache the access token.

### Phase 2: Execution
1.  **Connectivity Check:**
    Run `python connect_to_workspace.py`.
    *   *Verification:* Ensure the console reports "SUCCESS" and logs the connection latency.
2.  **Inventory Audit:**
    Run `python audit_model_inventory.py`.
    *   *Verification:* Review `logs/model_inventory.log` to confirm the availability of required classifiers.

### Phase 3: Maintenance
1.  **Cleanup:**
    Run `python 06_cleanup_local.py`.
    *   *Action:* This removes the `logs/` directory and recursively deletes all `__pycache__` folders.

## Appendix: Verified Algorithm Inventory

**Status:** Verified 2026-01-03
**Library Version:** scikit-learn v1.5.1
**Total Count:** 43 Algorithms

The `audit_model_inventory.py` script has confirmed the availability of the following classification algorithms in the local runtime. They are categorised below by mathematical family to distinguish between similar implementations (e.g., Single Trees vs Ensembles).

### Ensemble Methods
*Aggregates predictions from multiple base estimators to improve robustness.*
*   `AdaBoostClassifier`
*   `BaggingClassifier`
*   `ExtraTreesClassifier` (Ensemble of randomised trees)
*   `GradientBoostingClassifier`
*   `HistGradientBoostingClassifier` (Optimised for large datasets)
*   `RandomForestClassifier`
*   `StackingClassifier`
*   `VotingClassifier`

### Linear Models & Discriminant Analysis
*   `LogisticRegression`
*   `LogisticRegressionCV` (Built-in Cross-Validation)
*   `RidgeClassifier`
*   `RidgeClassifierCV`
*   `SGDClassifier` (Stochastic Gradient Descent)
*   `Perceptron`
*   `PassiveAggressiveClassifier`
*   `LinearDiscriminantAnalysis`
*   `QuadraticDiscriminantAnalysis`

### Support Vector Machines (SVM)
*   `LinearSVC`
*   `NuSVC`
*   `SVC`

### Naive Bayes
*   `BernoulliNB`
*   `CategoricalNB`
*   `ComplementNB`
*   `GaussianNB`
*   `MultinomialNB`

### Trees & Neighbors
*   `DecisionTreeClassifier`
*   `ExtraTreeClassifier` (Single randomised tree)
*   `KNeighborsClassifier`
*   `RadiusNeighborsClassifier`
*   `NearestCentroid`

### Meta-Estimators & Multi-Class Strategies
*   `CalibratedClassifierCV`
*   `ClassifierChain`
*   `MultiOutputClassifier`
*   `OneVsOneClassifier`
*   `OneVsRestClassifier`
*   `OutputCodeClassifier`

### Semi-Supervised & Other
*   `LabelPropagation`
*   `LabelSpreading`
*   `GaussianProcessClassifier`
*   `MLPClassifier` (Multi-layer Perceptron / Neural Network)
*   `DummyClassifier` (Baseline comparison)
*   `FixedThresholdClassifier`
*   `TunedThresholdClassifierCV`
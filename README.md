# ML_Labs: Machine Learning Architectures

**System Owner:** Shane Lee
**Licence:** MIT

## 1. Executive Summary

**ML_Labs** functions as a central repository for resource-efficient Machine Learning architectures. The system is designed to decouple memory consumption from dataset size.

The architectures address the technical constraint where standard in-memory model training fails when dataset volume exceeds physical RAM. By implementing **Out-of-Core Learning** and **Incremental Batch Processing**, these modules process high-volume datasets ($N > 10^7$) on commodity hardware or restricted cloud containers. The system employs a hybrid compute model. Computationally intensive gradient descent operations are offloaded to **Azure Machine Learning Compute Clusters**, while inference workloads are executed on local infrastructure to minimise cloud operational costs.

## 2. Architectural Standards

All modules within this repository adhere to engineering protocols designed to maintain bounded memory usage, fault tolerance, and auditability.

### 2.1. Hybrid Compute Model
*   **Training:** Workloads are submitted to **Azure Machine Learning Compute Clusters**. This leverages horizontal scaling for the computationally expensive training phases.
*   **Inference:** Scoring is executed on **Local Infrastructure** using iterative batch processing engines. This removes the dependency on persistent, billable cloud endpoints for batch predictions.

### 2.2. Container-Native Resource Governance
Standard Python libraries often identify Host OS memory rather than container limits. This frequently leads to process termination (OOMKilled) in Docker or Kubernetes environments.
*   **Cgroup Detection:** All modules utilise a shared library (`scalability_utils.py`) to inspect Linux Control Groups (`/sys/fs/cgroup/memory.max`). The system dynamically calculates batch sizes based on the hard limit imposed by the orchestrator.

### 2.3. Bounded Memory Architecture
*   **Constant Space Complexity ($O(1)$):** Inference engines process data in discrete chunks (e.g. 50,000 rows). Memory usage is determined by the batch size and remains constant regardless of the total dataset volume.
*   **Probabilistic Deduplication:** Streaming data is deduplicated using **Bloom Filters**. This allows for membership testing without the linear memory cost associated with storing seen identifiers.
*   **Memory Mapping:** Large model artifacts are loaded using `joblib` with `mmap_mode='r'`. This enables the operating system to map the file directly into virtual memory. It allows multiple worker processes to share physical memory pages.

### 2.4. Operational Resilience
*   **Dead Letter Queues (DLQ):** ETL pipelines isolate malformed records to a dedicated error file. This allows the primary data stream to continue processing despite schema violations.
*   **Asynchronous I/O:** Data generation scripts utilise asynchronous disk I/O to mitigate read/write latency during high-bandwidth downloads.
*   **Structured Logging:** Telemetry is captured in structured JSON or text formats within a dedicated `logs/` directory to facilitate automated auditing.

## 3. Module Portfolio

### 3.1. Classification Architectures
*Architectures designed to categorise entities into discrete classes based on input vectors.*

*   **[Credit_Default/](./Credit_Default)**
    *   **Domain:** FinTech / Banking
    *   **Objective:** Predict the probability of borrower default to inform credit risk decisions.
    *   **Output:** Probability Score (0-1).
    *   **Technical Focus:** Implements a **Streaming Ensemble** with **Nystroem Kernel Approximation**. This projects features into a higher-dimensional space to approximate non-linear decision boundaries using linear solvers.

*   **[Feedback_Classification/](./Feedback_Classification)**
    *   **Domain:** Customer Experience / NLP
    *   **Objective:** Classify unstructured text streams to identify sentiment drivers and priority alerts.
    *   **Output:** Binary Classification / Confidence Score.
    *   **Technical Focus:** Utilises **Stateless Feature Extraction** (`HashingVectorizer`) and **Online Classification** (`PassiveAggressiveClassifier`) to process infinite text streams with constant memory complexity.

*   **[Student_Success/](./Student_Success)**
    *   **Domain:** Higher Education
    *   **Objective:** Predict student outcomes (Dropout, Enrolled, Graduate) to enable early intervention.
    *   **Output:** Multi-Class Probability Distribution.
    *   **Technical Focus:** Utilises **Out-of-Core Learning** with a multi-class SGD classifier. The system employs a **Bloom Filter** to deduplicate streaming academic records with a fixed memory footprint.

### 3.2. Regression Architectures
*Architectures designed to predict continuous numerical values based on quantitative metrics.*

*   **[Graduate_Admissions/](./Graduate_Admissions)**
    *   **Domain:** Higher Education
    *   **Objective:** Estimate the probability of candidate admission based on standardised test scores (GRE, TOEFL) and CGPA.
    *   **Output:** Continuous Probability Score (0-1).
    *   **Technical Focus:** Features an **Incremental Learning** pipeline. The system dynamically adjusts batch sizes at runtime based on real-time container resource detection to maintain throughput on constrained hardware.

### 3.3. Operational Utilities
*Diagnostic tools designed to validate infrastructure and environment readiness.*

*   **[Utils/](./Utils)**
    *   **Domain:** DevOps / MLOps
    *   **Objective:** Validate Azure connectivity, authentication tokens, and library dependencies prior to deployment.
    *   **Output:** Diagnostic Logs / Boolean Status.
    *   **Technical Focus:** Implements **Identity Negotiation** via `DefaultAzureCredential` and **Dynamic Resource Introspection**. This ensures local execution environments maintain parity with cloud targets before capital-intensive resources are provisioned.

## 4. Standard Operating Procedure (SOP)

The following workflow defines the operational protocols established to maintain repository maintenance and traceability.

1.  **Configuration:** Define credentials and hyperparameters in `config.json`. Do not hardcode secrets in source files.
2.  **Pre-Flight:** Execute `Utils/connect_to_workspace.py` to verify the Azure context and authentication status.
3.  **Training:** Submit workloads via `01_submit_job.py`.
4.  **Inference:** Generate predictions via `04_local_model_scoring.py`. Use the `dry_run` flag for initial validation.
5.  **Sanitisation:** Execute `06_cleanup_local.py` post-operation. This removes temporary artifacts, logs, and bytecode to prevent storage bloat.
"""
Script Name: audit_model_inventory.py
Author: Shane Lee
Description: Performs a runtime inspection of the scikit-learn library. 
             Generates a complete manifest of available classification algorithms 
             by querying the estimator registry. Verifies that the local environment 
             possesses the necessary mathematical dependencies for modelling.
Key Outputs:
    - logs/model_inventory.log (Audit Record)
    - Console Output
"""

import sys
import sklearn
from sklearn.utils import all_estimators
import scalability_utils as utils

def main():
    # --- Initialization ---
    config = utils.load_configuration()
    logger = utils.setup_logger("ModelInventoryAudit", "model_inventory.log", config)

    # --- Environment Inspection ---
    logger.info(f"--- SCIKIT-LEARN MODEL INVENTORY (v{sklearn.__version__}) ---")
    logger.info(f"{'Algorithm Name':<40} | {'Type':<20}")
    logger.info("-" * 65)

    # Retrieve all valid estimators from the library.
    # Filters specifically for 'classifier' types to align with the business problem.
    estimators = all_estimators(type_filter='classifier')

    count = 0
    for name, class_ in estimators:
        # Broadened scope: Removed the restrictive 'if any(...)' filter.
        # Now cataloguing ALL available classifiers to allow comprehensive evaluation.
        logger.info(f"{name:<40} | Classifier")
        count += 1

    logger.info("-" * 65)
    logger.info(f"Total Audit: {count} algorithms available for deployment.")
    logger.info(f"Audit Log Saved to: logs/model_inventory.log")

if __name__ == "__main__":
    main()
"""Runtime Inspection of Scikit-Learn Estimators.

Purpose:
    Generates a complete manifest of available classification algorithms by
    querying the local estimator registry. This diagnostic ensures that the
    local environment possesses the necessary mathematical dependencies
    required for consistent cross-environment modelling.

Workflow:
    1. Queries `sklearn.utils.all_estimators` for type='classifier'.
    2. Formats the output into a readable, whitespace-padded table.
    3. Logs the manifest to both console and disk.

Author: Shane Lee
Licence: MIT
"""

import logging
import sys
from typing import Any, Dict, List, Tuple, Type

import sklearn
from sklearn.base import ClassifierMixin
from sklearn.utils import all_estimators

# --- Local Utils Import ---
import scalability_utils as utils


def audit_estimators() -> List[Tuple[str, Type[ClassifierMixin]]]:
    """Retrieves all classification estimators available in the local registry.

    Returns:
        List[Tuple[str, Type[ClassifierMixin]]]: A list of (name, class) tuples.
    """
    # type_filter="classifier" covers all standard classification algos
    return all_estimators(type_filter="classifier")


def main() -> None:
    """Main execution flow for library introspection."""
    # --- Initialization ---
    config: Dict[str, Any] = utils.load_config()
    logger: logging.Logger = utils.configure_logging(
        "ModelInventoryAudit", "model_inventory.log", config
    )

    # --- Table Formatting Constants ---
    COL_WIDTH_NAME: int = 45
    COL_WIDTH_CAT: int = 15
    DIVIDER_LEN: int = COL_WIDTH_NAME + COL_WIDTH_CAT + 5

    # --- Environment Inspection ---
    logger.info(f"--- SCIKIT-LEARN AUDIT (v{sklearn.__version__}) ---")
    logger.info(f"{'Algorithm Name':<{COL_WIDTH_NAME}} | {'Category':<{COL_WIDTH_CAT}}")
    logger.info("-" * DIVIDER_LEN)

    try:
        estimators: List[Tuple[str, Type[ClassifierMixin]]] = audit_estimators()
        count: int = 0

        for name, _ in estimators:
            logger.info(f"{name:<{COL_WIDTH_NAME}} | Classifier")
            count += 1

        logger.info("-" * DIVIDER_LEN)
        logger.info(f"AUDIT COMPLETE: {count} algorithms available for deployment.")
        logger.info(f"Ledger saved to: {utils.ProjectPaths.LOGS}/model_inventory.log")

    except Exception as e:
        logger.error(f"CRITICAL: Audit failed due to environment corruption. {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
"""Email MLOps — Experiment Tracking & Infrastructure Utilities.

Provides a clean, backend-agnostic interface for experiment tracking.
Supports MLflow for production and a local JSON logger for offline development.
"""

from email_mlops.experiment_tracking.factory import create_tracker
from email_mlops.experiment_tracking.base import BaseExperimentTracker

__all__ = [
    "create_tracker",
    "BaseExperimentTracker",
]

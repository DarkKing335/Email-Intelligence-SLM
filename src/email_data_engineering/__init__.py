"""Email Data Engineering Module.

A clean-architecture, production-ready pipeline package for fine-tuning SLM datasets.
"""

from email_data_engineering.pipeline import DataEngineeringPipeline
from email_data_engineering.logging_config import configure_logging

__version__ = "0.1.0"
__all__ = [
    "DataEngineeringPipeline",
    "configure_logging",
]

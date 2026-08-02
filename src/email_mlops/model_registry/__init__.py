"""Model Registry (US-6.4).

Backend-agnostic storage for approved model artifacts (LoRA adapters /
checkpoints) with metadata, lineage, and multi-version management.
"""

from email_mlops.model_registry.base import (
    BaseModelRegistry,
    ModelVersion,
    RegistryStage,
)
from email_mlops.model_registry.factory import create_registry
from email_mlops.model_registry.local_registry import LocalModelRegistry

__all__ = [
    "BaseModelRegistry",
    "ModelVersion",
    "RegistryStage",
    "LocalModelRegistry",
    "create_registry",
]

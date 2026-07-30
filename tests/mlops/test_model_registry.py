"""Unit tests for the Model Registry module (US-6.4)."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from email_mlops.model_registry import (
    LocalModelRegistry,
    RegistryStage,
    create_registry,
)

# ── LocalModelRegistry ────────────────────────────────────────────────────────


class TestLocalModelRegistry:
    def test_register_creates_version_one(self, tmp_dir: Path) -> None:
        reg = LocalModelRegistry(root_dir=tmp_dir)
        entry = reg.register_model(
            "reply-style-professional",
            adapter_path="models/adapters/professional",
            source_run_id="run-123",
            base_model="Qwen3-4B",
            dataset_version="v0.1.0",
            metrics={"val_loss": 0.42},
            description="First professional adapter.",
        )

        assert entry.version == 1
        assert entry.stage == RegistryStage.NONE
        assert entry.base_model == "Qwen3-4B"
        assert entry.dataset_version == "v0.1.0"
        assert entry.source_run_id == "run-123"
        assert entry.metrics["val_loss"] == 0.42

        # Persisted to disk
        assert (tmp_dir / "reply-style-professional.json").exists()

    def test_versions_increment_and_persist(self, tmp_dir: Path) -> None:
        reg = LocalModelRegistry(root_dir=tmp_dir)
        reg.register_model("friendly", adapter_path="p1")
        reg.register_model("friendly", adapter_path="p2")
        v3 = reg.register_model("friendly", adapter_path="p3")

        assert v3.version == 3

        # A fresh registry instance reads the same persisted state
        reg2 = LocalModelRegistry(root_dir=tmp_dir)
        versions = reg2.list_versions("friendly")
        assert [v.version for v in versions] == [1, 2, 3]
        assert versions[2].adapter_path == "p3"

    def test_list_models(self, tmp_dir: Path) -> None:
        reg = LocalModelRegistry(root_dir=tmp_dir)
        reg.register_model("concise", adapter_path="a")
        reg.register_model("professional", adapter_path="b")
        assert reg.list_models() == ["concise", "professional"]

    def test_get_version_and_missing(self, tmp_dir: Path) -> None:
        reg = LocalModelRegistry(root_dir=tmp_dir)
        reg.register_model("concise", adapter_path="a")
        assert reg.get_version("concise", 1).adapter_path == "a"
        with pytest.raises(KeyError):
            reg.get_version("concise", 99)

    def test_promote_to_production_archives_previous(self, tmp_dir: Path) -> None:
        reg = LocalModelRegistry(root_dir=tmp_dir)
        reg.register_model("concise", adapter_path="a")  # v1
        reg.register_model("concise", adapter_path="b")  # v2

        reg.transition_stage("concise", 1, RegistryStage.PRODUCTION)
        prod = reg.get_deployment_target("concise")
        assert prod is not None and prod.version == 1

        # Promoting v2 must archive v1
        reg.transition_stage("concise", 2, RegistryStage.PRODUCTION)
        assert reg.get_version("concise", 1).stage == RegistryStage.ARCHIVED
        assert reg.get_version("concise", 2).stage == RegistryStage.PRODUCTION

        target = reg.get_deployment_target("concise")
        assert target is not None and target.version == 2

    def test_deployment_target_prefers_production_then_staging(self, tmp_dir: Path) -> None:
        reg = LocalModelRegistry(root_dir=tmp_dir)
        reg.register_model("concise", adapter_path="a")  # v1
        reg.register_model("concise", adapter_path="b")  # v2

        # No production yet → falls back to latest staging
        reg.transition_stage("concise", 1, RegistryStage.STAGING)
        target = reg.get_deployment_target("concise")
        assert target is not None and target.version == 1

        # Production wins over staging
        reg.transition_stage("concise", 2, RegistryStage.PRODUCTION)
        target = reg.get_deployment_target("concise")
        assert target is not None and target.version == 2

    def test_deployment_target_none_when_empty(self, tmp_dir: Path) -> None:
        reg = LocalModelRegistry(root_dir=tmp_dir)
        assert reg.get_deployment_target("nonexistent") is None
        assert reg.list_versions("nonexistent") == []

    def test_get_latest_version_with_stage_filter(self, tmp_dir: Path) -> None:
        reg = LocalModelRegistry(root_dir=tmp_dir)
        reg.register_model("concise", adapter_path="a")  # v1
        reg.register_model("concise", adapter_path="b")  # v2
        reg.transition_stage("concise", 1, RegistryStage.STAGING)

        assert reg.get_latest_version("concise").version == 2
        assert reg.get_latest_version("concise", stage=RegistryStage.STAGING).version == 1
        assert reg.get_latest_version("concise", stage=RegistryStage.PRODUCTION) is None

    def test_invalid_name_rejected(self, tmp_dir: Path) -> None:
        reg = LocalModelRegistry(root_dir=tmp_dir)
        with pytest.raises(ValueError, match="Invalid model name"):
            reg.register_model("bad name/with slash", adapter_path="a")

    def test_transition_unregistered_raises(self, tmp_dir: Path) -> None:
        reg = LocalModelRegistry(root_dir=tmp_dir)
        with pytest.raises(KeyError):
            reg.transition_stage("ghost", 1, RegistryStage.STAGING)


# ── Factory ───────────────────────────────────────────────────────────────────


class TestRegistryFactory:
    def test_defaults_to_local(self, tmp_dir: Path) -> None:
        config_path = tmp_dir / "mlops.yaml"
        config_path.write_text(
            "model_registry:\n"
            "  backend: local\n"
            "  local:\n"
            f"    root_dir: '{tmp_dir / 'registry'}'\n",
            encoding="utf-8",
        )
        reg = create_registry(config_path=config_path)
        assert isinstance(reg, LocalModelRegistry)

    def test_missing_config_defaults_to_local(self) -> None:
        reg = create_registry(config_path="/nonexistent/mlops.yaml")
        assert isinstance(reg, LocalModelRegistry)

    def test_backend_override_argument(self, tmp_dir: Path) -> None:
        config_path = tmp_dir / "mlops.yaml"
        config_path.write_text("model_registry:\n  backend: mlflow\n", encoding="utf-8")
        reg = create_registry(config_path=config_path, backend_override="local")
        assert isinstance(reg, LocalModelRegistry)

    def test_env_var_override(self, tmp_dir: Path) -> None:
        config_path = tmp_dir / "mlops.yaml"
        config_path.write_text("model_registry:\n  backend: mlflow\n", encoding="utf-8")
        with patch.dict(os.environ, {"MODEL_REGISTRY_BACKEND": "local"}):
            reg = create_registry(config_path=config_path)
        assert isinstance(reg, LocalModelRegistry)

    def test_unknown_backend_raises(self, tmp_dir: Path) -> None:
        config_path = tmp_dir / "mlops.yaml"
        config_path.write_text("model_registry:\n  backend: sagemaker\n", encoding="utf-8")
        with pytest.raises(ValueError, match="Unknown model registry backend"):
            create_registry(config_path=config_path)


# ── MlflowModelRegistry (mocked) ──────────────────────────────────────────────


class TestMlflowModelRegistry:
    @patch("email_mlops.model_registry.mlflow_registry.MlflowClient")
    @patch("email_mlops.model_registry.mlflow_registry.mlflow")
    def test_register_maps_lineage_to_tags(
        self, mock_mlflow: MagicMock, mock_client_cls: MagicMock
    ) -> None:
        from email_mlops.model_registry.mlflow_registry import MlflowModelRegistry

        client = mock_client_cls.return_value
        returned = MagicMock()
        returned.name = "professional"
        returned.version = "1"
        returned.current_stage = "None"
        returned.source = "models/adapters/professional"
        returned.run_id = "run-123"
        returned.tags = {
            "lineage.base_model": "Qwen3-4B",
            "lineage.dataset_version": "v0.1.0",
            "metric.val_loss": "0.42",
        }
        returned.description = ""
        client.create_model_version.return_value = returned

        reg = MlflowModelRegistry(tracking_uri="http://test:5000")
        entry = reg.register_model(
            "professional",
            adapter_path="models/adapters/professional",
            source_run_id="run-123",
            base_model="Qwen3-4B",
            dataset_version="v0.1.0",
            metrics={"val_loss": 0.42},
        )

        # Lineage + metrics were forwarded as tags to MLflow
        _, kwargs = client.create_model_version.call_args
        assert kwargs["tags"]["lineage.base_model"] == "Qwen3-4B"
        assert kwargs["tags"]["lineage.dataset_version"] == "v0.1.0"
        assert kwargs["tags"]["metric.val_loss"] == "0.42"

        # And round-trip back into our ModelVersion abstraction
        assert entry.base_model == "Qwen3-4B"
        assert entry.dataset_version == "v0.1.0"
        assert entry.metrics["val_loss"] == 0.42
        assert entry.stage == RegistryStage.NONE

    @patch("email_mlops.model_registry.mlflow_registry.MlflowClient")
    @patch("email_mlops.model_registry.mlflow_registry.mlflow")
    def test_promote_archives_existing(
        self, mock_mlflow: MagicMock, mock_client_cls: MagicMock
    ) -> None:
        from email_mlops.model_registry.mlflow_registry import MlflowModelRegistry

        client = mock_client_cls.return_value
        returned = MagicMock()
        returned.name = "professional"
        returned.version = "2"
        returned.current_stage = "Production"
        returned.source = "p2"
        returned.run_id = ""
        returned.tags = {}
        returned.description = ""
        client.transition_model_version_stage.return_value = returned

        reg = MlflowModelRegistry(tracking_uri="http://test:5000")
        reg.transition_stage("professional", 2, RegistryStage.PRODUCTION)

        _, kwargs = client.transition_model_version_stage.call_args
        assert kwargs["stage"] == "Production"
        assert kwargs["archive_existing_versions"] is True

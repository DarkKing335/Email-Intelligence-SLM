"""Unit tests for the Experiment Tracking module (US-6.3)."""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from email_mlops.experiment_tracking.base import BaseExperimentTracker
from email_mlops.experiment_tracking.local_tracker import LocalTracker
from email_mlops.experiment_tracking.mlflow_tracker import MlflowTracker
from email_mlops.experiment_tracking.factory import create_tracker


# ── LocalTracker Tests ────────────────────────────────────────────────────────

class TestLocalTracker:
    """Tests for the local JSON file tracker."""

    def test_start_and_end_run_writes_json(self, tmp_dir: Path) -> None:
        tracker = LocalTracker(output_dir=tmp_dir)
        run_id = tracker.start_run(
            run_name="test-run-1",
            experiment_name="unit-test",
            tags={"env": "test"},
        )

        assert run_id is not None
        assert tracker.get_run_id() == run_id

        tracker.end_run(status="FINISHED")

        # File should exist
        json_path = tmp_dir / f"{run_id}.json"
        assert json_path.exists()

        # Validate content
        with json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        assert data["run_id"] == run_id
        assert data["run_name"] == "test-run-1"
        assert data["experiment_name"] == "unit-test"
        assert data["status"] == "FINISHED"
        assert data["tags"]["env"] == "test"
        assert data["start_time"] is not None
        assert data["end_time"] is not None

    def test_log_parameters(self, tmp_dir: Path) -> None:
        tracker = LocalTracker(output_dir=tmp_dir)
        run_id = tracker.start_run(run_name="param-test")

        tracker.log_parameter("learning_rate", 0.001)
        tracker.log_parameter("epochs", 10)
        tracker.log_parameters({"batch_size": 32, "model": "phi-3-mini"})

        tracker.end_run()

        with (tmp_dir / f"{run_id}.json").open("r") as f:
            data = json.load(f)

        assert data["parameters"]["learning_rate"] == "0.001"
        assert data["parameters"]["epochs"] == "10"
        assert data["parameters"]["batch_size"] == "32"
        assert data["parameters"]["model"] == "phi-3-mini"

    def test_log_metrics_with_steps(self, tmp_dir: Path) -> None:
        tracker = LocalTracker(output_dir=tmp_dir)
        run_id = tracker.start_run(run_name="metric-test")

        tracker.log_metric("train_loss", 0.95, step=1)
        tracker.log_metric("train_loss", 0.72, step=2)
        tracker.log_metric("train_loss", 0.41, step=3)
        tracker.log_metrics({"val_loss": 0.55, "val_accuracy": 0.88}, step=3)

        tracker.end_run()

        with (tmp_dir / f"{run_id}.json").open("r") as f:
            data = json.load(f)

        metrics = data["metrics"]
        assert len(metrics) == 5

        # Check step ordering
        loss_entries = [m for m in metrics if m["key"] == "train_loss"]
        assert len(loss_entries) == 3
        assert loss_entries[0]["step"] == 1
        assert loss_entries[2]["step"] == 3
        assert loss_entries[2]["value"] == 0.41

    def test_log_artifact(self, tmp_dir: Path) -> None:
        tracker = LocalTracker(output_dir=tmp_dir)
        run_id = tracker.start_run(run_name="artifact-test")

        tracker.log_artifact("/path/to/model.bin", artifact_path="models/")
        tracker.log_artifact("/path/to/metrics.json")

        tracker.end_run()

        with (tmp_dir / f"{run_id}.json").open("r") as f:
            data = json.load(f)

        assert len(data["artifacts"]) == 2
        assert data["artifacts"][0]["local_path"] == "/path/to/model.bin"
        assert data["artifacts"][0]["artifact_path"] == "models/"
        assert data["artifacts"][1]["artifact_path"] is None

    def test_set_tags(self, tmp_dir: Path) -> None:
        tracker = LocalTracker(output_dir=tmp_dir)
        run_id = tracker.start_run(
            run_name="tag-test",
            tags={"initial": "true"},
        )

        tracker.set_tags({"phase": "training", "version": "v0.1.0"})

        tracker.end_run()

        with (tmp_dir / f"{run_id}.json").open("r") as f:
            data = json.load(f)

        assert data["tags"]["initial"] == "true"
        assert data["tags"]["phase"] == "training"
        assert data["tags"]["version"] == "v0.1.0"

    def test_end_run_clears_state(self, tmp_dir: Path) -> None:
        tracker = LocalTracker(output_dir=tmp_dir)
        tracker.start_run(run_name="clear-test")
        assert tracker.get_run_id() is not None

        tracker.end_run()
        assert tracker.get_run_id() is None

    def test_list_runs(self, tmp_dir: Path) -> None:
        tracker = LocalTracker(output_dir=tmp_dir)

        # Create two runs
        tracker.start_run(run_name="run-1")
        tracker.log_parameter("lr", 0.001)
        tracker.log_metric("loss", 0.5)
        tracker.end_run()

        tracker.start_run(run_name="run-2")
        tracker.log_metric("loss", 0.3)
        tracker.log_metric("acc", 0.9)
        tracker.end_run()

        runs = tracker.list_runs()
        assert len(runs) == 2
        assert runs[0]["run_name"] == "run-1"
        assert runs[0]["num_parameters"] == 1
        assert runs[0]["num_metrics"] == 1
        assert runs[1]["run_name"] == "run-2"
        assert runs[1]["num_metrics"] == 2

    def test_interrupted_run_on_new_start(self, tmp_dir: Path) -> None:
        tracker = LocalTracker(output_dir=tmp_dir)
        first_id = tracker.start_run(run_name="first")

        # Starting a new run should close the old one as INTERRUPTED
        second_id = tracker.start_run(run_name="second")

        assert tracker.get_run_id() == second_id
        assert first_id != second_id

        # First run file should exist with INTERRUPTED status
        with (tmp_dir / f"{first_id}.json").open("r") as f:
            data = json.load(f)
        assert data["status"] == "INTERRUPTED"

        tracker.end_run()


# ── MlflowTracker Tests (Mocked) ─────────────────────────────────────────────

class TestMlflowTracker:
    """Tests for the MLflow tracker using mocked mlflow SDK calls."""

    @patch("email_mlops.experiment_tracking.mlflow_tracker.mlflow")
    def test_start_run(self, mock_mlflow: MagicMock) -> None:
        # Setup mock
        mock_experiment = MagicMock()
        mock_experiment.experiment_id = "exp-123"
        mock_mlflow.get_experiment_by_name.return_value = mock_experiment

        mock_run = MagicMock()
        mock_run.info.run_id = "run-abc-456"
        mock_mlflow.start_run.return_value = mock_run

        tracker = MlflowTracker(
            tracking_uri="http://test:5000",
            default_experiment="test-experiment",
        )
        run_id = tracker.start_run(run_name="test-run", tags={"env": "ci"})

        assert run_id == "run-abc-456"
        mock_mlflow.start_run.assert_called_once_with(
            run_name="test-run",
            experiment_id="exp-123",
            tags={"env": "ci"},
        )

    @patch("email_mlops.experiment_tracking.mlflow_tracker.mlflow")
    def test_log_parameter(self, mock_mlflow: MagicMock) -> None:
        tracker = MlflowTracker(tracking_uri="http://test:5000")
        tracker._run_id = "mock-run"

        tracker.log_parameter("lr", 0.001)
        mock_mlflow.log_param.assert_called_once_with("lr", 0.001)

    @patch("email_mlops.experiment_tracking.mlflow_tracker.mlflow")
    def test_log_parameters_batch(self, mock_mlflow: MagicMock) -> None:
        tracker = MlflowTracker(tracking_uri="http://test:5000")
        tracker._run_id = "mock-run"

        params = {"lr": 0.001, "epochs": 5}
        tracker.log_parameters(params)
        mock_mlflow.log_params.assert_called_once_with(params)

    @patch("email_mlops.experiment_tracking.mlflow_tracker.mlflow")
    def test_log_metric(self, mock_mlflow: MagicMock) -> None:
        tracker = MlflowTracker(tracking_uri="http://test:5000")
        tracker._run_id = "mock-run"

        tracker.log_metric("loss", 0.42, step=5)
        mock_mlflow.log_metric.assert_called_once_with("loss", 0.42, step=5)

    @patch("email_mlops.experiment_tracking.mlflow_tracker.mlflow")
    def test_log_metrics_batch(self, mock_mlflow: MagicMock) -> None:
        tracker = MlflowTracker(tracking_uri="http://test:5000")
        tracker._run_id = "mock-run"

        metrics = {"loss": 0.42, "acc": 0.91}
        tracker.log_metrics(metrics, step=5)
        mock_mlflow.log_metrics.assert_called_once_with(metrics, step=5)

    @patch("email_mlops.experiment_tracking.mlflow_tracker.mlflow")
    def test_log_artifact(self, mock_mlflow: MagicMock) -> None:
        tracker = MlflowTracker(tracking_uri="http://test:5000")
        tracker._run_id = "mock-run"

        tracker.log_artifact("/tmp/model.bin", artifact_path="models/")
        mock_mlflow.log_artifact.assert_called_once_with(
            "/tmp/model.bin", artifact_path="models/"
        )

    @patch("email_mlops.experiment_tracking.mlflow_tracker.mlflow")
    def test_set_tags(self, mock_mlflow: MagicMock) -> None:
        tracker = MlflowTracker(tracking_uri="http://test:5000")
        tracker._run_id = "mock-run"

        tracker.set_tags({"version": "v1.0", "env": "prod"})
        mock_mlflow.set_tags.assert_called_once_with(
            {"version": "v1.0", "env": "prod"}
        )

    @patch("email_mlops.experiment_tracking.mlflow_tracker.mlflow")
    def test_end_run(self, mock_mlflow: MagicMock) -> None:
        tracker = MlflowTracker(tracking_uri="http://test:5000")
        tracker._run_id = "mock-run"

        tracker.end_run(status="FINISHED")
        mock_mlflow.end_run.assert_called_once_with(status="FINISHED")
        assert tracker.get_run_id() is None


# ── Factory Tests ─────────────────────────────────────────────────────────────

class TestTrackerFactory:
    """Tests for the tracker factory selection logic."""

    def test_factory_creates_local_tracker_by_default(self, tmp_dir: Path) -> None:
        # Write a minimal config with backend: local
        config_path = tmp_dir / "mlops.yaml"
        config_path.write_text(
            "tracking:\n"
            "  backend: local\n"
            "  local:\n"
            f"    output_dir: '{tmp_dir / 'experiments'}'\n"
            "    pretty_print: true\n",
            encoding="utf-8",
        )

        tracker = create_tracker(config_path=config_path)
        assert isinstance(tracker, LocalTracker)

    @patch("email_mlops.experiment_tracking.mlflow_tracker.mlflow")
    def test_factory_creates_mlflow_tracker(
        self, mock_mlflow: MagicMock, tmp_dir: Path
    ) -> None:
        config_path = tmp_dir / "mlops.yaml"
        config_path.write_text(
            "tracking:\n"
            "  backend: mlflow\n"
            "  mlflow:\n"
            "    tracking_uri: 'http://test:5000'\n"
            "    default_experiment: 'test-exp'\n",
            encoding="utf-8",
        )

        tracker = create_tracker(config_path=config_path)
        assert isinstance(tracker, MlflowTracker)

    def test_factory_env_var_override(self, tmp_dir: Path) -> None:
        config_path = tmp_dir / "mlops.yaml"
        config_path.write_text(
            "tracking:\n"
            "  backend: mlflow\n"  # Config says mlflow
            "  local:\n"
            f"    output_dir: '{tmp_dir / 'experiments'}'\n",
            encoding="utf-8",
        )

        # Environment variable overrides to local
        with patch.dict(os.environ, {"TRACKING_BACKEND": "local"}):
            tracker = create_tracker(config_path=config_path)
        assert isinstance(tracker, LocalTracker)

    def test_factory_backend_override_argument(self, tmp_dir: Path) -> None:
        config_path = tmp_dir / "mlops.yaml"
        config_path.write_text(
            "tracking:\n"
            "  backend: mlflow\n"  # Config says mlflow
            "  local:\n"
            f"    output_dir: '{tmp_dir / 'experiments'}'\n",
            encoding="utf-8",
        )

        # Argument override takes highest priority
        tracker = create_tracker(
            config_path=config_path, backend_override="local"
        )
        assert isinstance(tracker, LocalTracker)

    def test_factory_unknown_backend_raises(self, tmp_dir: Path) -> None:
        config_path = tmp_dir / "mlops.yaml"
        config_path.write_text(
            "tracking:\n"
            "  backend: wandb\n",
            encoding="utf-8",
        )

        with pytest.raises(ValueError, match="Unknown tracking backend"):
            create_tracker(config_path=config_path)

    def test_factory_missing_config_defaults_to_local(self) -> None:
        # Point to a non-existent config → should fall back to local
        tracker = create_tracker(config_path="/nonexistent/path/mlops.yaml")
        assert isinstance(tracker, LocalTracker)

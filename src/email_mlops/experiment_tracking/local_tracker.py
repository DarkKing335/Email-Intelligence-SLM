"""Local JSON file experiment tracker.

Offline fallback that writes experiment data to structured JSON files.
Useful for development, CI testing, and air-gapped environments where
an MLflow server is unavailable.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from loguru import logger

from email_mlops.experiment_tracking.base import BaseExperimentTracker


class LocalTracker(BaseExperimentTracker):
    """Experiment tracker that persists runs to local JSON files.

    Output structure::

        {output_dir}/
          {run_id}.json     ← one file per run

    Each JSON file contains::

        {
            "run_id": "...",
            "run_name": "...",
            "experiment_name": "...",
            "status": "FINISHED",
            "start_time": "...",
            "end_time": "...",
            "tags": { ... },
            "parameters": { ... },
            "metrics": [ { "key": "...", "value": ..., "step": ..., "timestamp": "..." } ],
            "artifacts": [ ... ]
        }

    Parameters
    ----------
    output_dir:
        Directory where JSON run files are written.
    pretty_print:
        If ``True``, JSON files are indented for human readability.
    """

    def __init__(
        self,
        output_dir: str | Path = "reports/experiments",
        pretty_print: bool = True,
    ) -> None:
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._pretty_print = pretty_print

        # Active run state
        self._run_id: str | None = None
        self._run_data: dict[str, Any] = {}

        logger.info(
            f"[LocalTracker] Initialised — output dir: {self._output_dir}"
        )

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start_run(
        self,
        run_name: str,
        experiment_name: str | None = None,
        tags: dict[str, str] | None = None,
    ) -> str:
        # End any lingering run
        if self._run_id is not None:
            self.end_run(status="INTERRUPTED")

        self._run_id = str(uuid.uuid4())
        self._run_data = {
            "run_id": self._run_id,
            "run_name": run_name,
            "experiment_name": experiment_name or "default",
            "status": "RUNNING",
            "start_time": datetime.now(timezone.utc).isoformat(),
            "end_time": None,
            "tags": dict(tags) if tags else {},
            "parameters": {},
            "metrics": [],
            "artifacts": [],
        }

        logger.info(
            f"[LocalTracker] Started run '{run_name}' "
            f"(run_id={self._run_id}, experiment={self._run_data['experiment_name']})"
        )
        return self._run_id

    def end_run(self, status: str = "FINISHED") -> None:
        if self._run_id is None:
            logger.warning("[LocalTracker] end_run called with no active run.")
            return

        self._run_data["status"] = status
        self._run_data["end_time"] = datetime.now(timezone.utc).isoformat()

        # Write JSON file
        output_path = self._output_dir / f"{self._run_id}.json"
        indent = 2 if self._pretty_print else None
        with output_path.open("w", encoding="utf-8") as fh:
            json.dump(self._run_data, fh, indent=indent, ensure_ascii=False)
            fh.write("\n")

        logger.info(
            f"[LocalTracker] Ended run {self._run_id} with status={status}. "
            f"Saved to {output_path}"
        )
        self._run_id = None
        self._run_data = {}

    def get_run_id(self) -> str | None:
        return self._run_id

    # ── Parameters ────────────────────────────────────────────────────────────

    def log_parameter(self, key: str, value: Any) -> None:
        if self._run_id is None:
            logger.warning("[LocalTracker] log_parameter called with no active run.")
            return
        self._run_data["parameters"][key] = str(value)

    # ── Metrics ───────────────────────────────────────────────────────────────

    def log_metric(
        self,
        key: str,
        value: float,
        step: int | None = None,
    ) -> None:
        if self._run_id is None:
            logger.warning("[LocalTracker] log_metric called with no active run.")
            return
        entry = {
            "key": key,
            "value": value,
            "step": step,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self._run_data["metrics"].append(entry)

    # ── Artifacts ─────────────────────────────────────────────────────────────

    def log_artifact(
        self,
        local_path: str,
        artifact_path: str | None = None,
    ) -> None:
        if self._run_id is None:
            logger.warning("[LocalTracker] log_artifact called with no active run.")
            return
        self._run_data["artifacts"].append({
            "local_path": local_path,
            "artifact_path": artifact_path,
            "logged_at": datetime.now(timezone.utc).isoformat(),
        })
        logger.debug(f"[LocalTracker] Recorded artifact ref: {local_path}")

    # ── Tags ──────────────────────────────────────────────────────────────────

    def set_tags(self, tags: dict[str, str]) -> None:
        if self._run_id is None:
            logger.warning("[LocalTracker] set_tags called with no active run.")
            return
        self._run_data["tags"].update(tags)

    # ── Utility ───────────────────────────────────────────────────────────────

    def list_runs(self) -> list[dict[str, Any]]:
        """Load and return all completed run summaries from the output directory."""
        runs: list[dict[str, Any]] = []
        for json_file in sorted(self._output_dir.glob("*.json")):
            with json_file.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
                runs.append({
                    "run_id": data.get("run_id"),
                    "run_name": data.get("run_name"),
                    "experiment_name": data.get("experiment_name"),
                    "status": data.get("status"),
                    "start_time": data.get("start_time"),
                    "end_time": data.get("end_time"),
                    "num_parameters": len(data.get("parameters", {})),
                    "num_metrics": len(data.get("metrics", [])),
                    "num_artifacts": len(data.get("artifacts", [])),
                })
        return runs

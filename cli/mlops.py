"""CLI entrypoint for MLOps operations (Epic 6).

Commands for the Model Registry (US-6.4), Health Checks (US-6.6), and
Deployment (US-6.5):

    email-mlops register-model NAME --adapter-path PATH [...]
    email-mlops list-models
    email-mlops list-versions NAME
    email-mlops promote-model NAME VERSION --stage production
    email-mlops deployment-target NAME
    email-mlops health [--live] [--deps mlflow,postgres]
    email-mlops environments
    email-mlops deploy --env staging [--dry-run] [--build/--no-build]
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import click
import yaml
from rich.console import Console
from rich.table import Table

from email_mlops.deployment import Deployer, DeploymentStatus, list_environments
from email_mlops.health import HealthChecker, HealthStatus, build_checker
from email_mlops.model_registry import RegistryStage, create_registry

console = Console()

DEFAULT_CONFIG = Path("configs/mlops.yaml")


def _load_config(config_path: Path) -> dict[str, Any]:
    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}
    return {}


@click.group()
@click.option(
    "--config",
    "-c",
    type=click.Path(dir_okay=False, path_type=Path),
    default=DEFAULT_CONFIG,
    show_default=True,
    help="Path to the MLOps YAML configuration file.",
)
@click.pass_context
def cli(ctx: click.Context, config: Path) -> None:
    """Email Intelligence SLM — MLOps operations (Epic 6)."""
    ctx.ensure_object(dict)
    ctx.obj["CONFIG_PATH"] = config
    ctx.obj["CONFIG"] = _load_config(config)


# ── US-6.4 Model Registry ─────────────────────────────────────────────────────


@cli.command(name="register-model")
@click.argument("name", type=str)
@click.option("--adapter-path", required=True, help="Path/URI of the adapter or checkpoint.")
@click.option("--run-id", default=None, help="Source experiment run id (lineage).")
@click.option("--base-model", default=None, help="Base model, e.g. 'Qwen3-4B' (lineage).")
@click.option("--dataset-version", default=None, help="Training dataset version (lineage).")
@click.option(
    "--metric",
    "metrics",
    multiple=True,
    help="Metric as key=value (repeatable), e.g. --metric val_loss=0.42.",
)
@click.option("--description", default=None, help="Human-readable notes.")
@click.pass_context
def register_model(
    ctx: click.Context,
    name: str,
    adapter_path: str,
    run_id: str | None,
    base_model: str | None,
    dataset_version: str | None,
    metrics: tuple[str, ...],
    description: str | None,
) -> None:
    """US-6.4: Register an approved model artifact as a new version."""
    parsed_metrics: dict[str, float] = {}
    for item in metrics:
        if "=" not in item:
            raise click.BadParameter(f"Metric '{item}' must be key=value.")
        key, value = item.split("=", 1)
        try:
            parsed_metrics[key.strip()] = float(value)
        except ValueError as exc:
            raise click.BadParameter(f"Metric '{key}' value must be numeric.") from exc

    registry = create_registry(config_path=ctx.obj["CONFIG_PATH"])
    entry = registry.register_model(
        name,
        adapter_path,
        source_run_id=run_id,
        base_model=base_model,
        dataset_version=dataset_version,
        metrics=parsed_metrics,
        description=description,
    )
    console.print(
        f"[bold green]✔ Registered '{entry.name}' version {entry.version}[/] "
        f"(stage={entry.stage.value})"
    )
    console.print_json(data=entry.model_dump(mode="json"))


@cli.command(name="list-models")
@click.pass_context
def list_models(ctx: click.Context) -> None:
    """US-6.4: List all registered model names."""
    registry = create_registry(config_path=ctx.obj["CONFIG_PATH"])
    names = registry.list_models()
    if not names:
        console.print("[yellow]No models registered.[/]")
        return
    for name in names:
        latest = registry.get_latest_version(name)
        version = latest.version if latest else "?"
        console.print(f"  • [cyan]{name}[/]  (latest: v{version})")


@cli.command(name="list-versions")
@click.argument("name", type=str)
@click.pass_context
def list_versions(ctx: click.Context, name: str) -> None:
    """US-6.4: List every registered version of a model."""
    registry = create_registry(config_path=ctx.obj["CONFIG_PATH"])
    versions = registry.list_versions(name)
    if not versions:
        console.print(f"[yellow]No versions found for '{name}'.[/]")
        return

    table = Table(title=f"Versions of '{name}'")
    table.add_column("Version", style="cyan", justify="right")
    table.add_column("Stage", style="magenta")
    table.add_column("Base Model", style="green")
    table.add_column("Dataset", style="green")
    table.add_column("Run ID", style="blue")
    table.add_column("Created", style="dim")
    for v in versions:
        table.add_row(
            str(v.version),
            v.stage.value,
            v.base_model or "—",
            v.dataset_version or "—",
            (v.source_run_id or "—")[:12],
            v.created_at.split("T")[0],
        )
    console.print(table)


@cli.command(name="promote-model")
@click.argument("name", type=str)
@click.argument("version", type=int)
@click.option(
    "--stage",
    type=click.Choice([s.value for s in RegistryStage]),
    required=True,
    help="Target lifecycle stage.",
)
@click.pass_context
def promote_model(ctx: click.Context, name: str, version: int, stage: str) -> None:
    """US-6.4: Transition a version's stage (select a version to deploy)."""
    registry = create_registry(config_path=ctx.obj["CONFIG_PATH"])
    try:
        entry = registry.transition_stage(name, version, RegistryStage(stage))
    except KeyError as exc:
        console.print(f"[bold red]Error:[/] {exc}")
        raise click.Abort()
    console.print(f"[bold green]✔ '{entry.name}' version {entry.version} → {entry.stage.value}[/]")


@cli.command(name="deployment-target")
@click.argument("name", type=str)
@click.pass_context
def deployment_target(ctx: click.Context, name: str) -> None:
    """US-6.4: Show which version deployment tooling would ship for a model."""
    registry = create_registry(config_path=ctx.obj["CONFIG_PATH"])
    target = registry.get_deployment_target(name)
    if target is None:
        console.print(f"[yellow]No deployable version for '{name}'.[/]")
        return
    console.print(
        f"[bold green]Deployment target for '{name}':[/] "
        f"version {target.version} (stage={target.stage.value})"
    )
    console.print_json(data=target.model_dump(mode="json"))


# ── US-6.6 Health Check ───────────────────────────────────────────────────────


def _build_checker_from_config(
    cfg: dict[str, Any],
    only_deps: set[str] | None = None,
) -> HealthChecker:
    """Construct a :class:`HealthChecker` from the ``health`` config block."""
    health_cfg = cfg.get("health", {})
    return build_checker(
        service=health_cfg.get("service", "email-intelligence-slm"),
        dependencies=health_cfg.get("dependencies"),
        only=only_deps,
    )


@cli.command(name="health")
@click.option("--live", is_flag=True, help="Run a liveness check only (no dependencies).")
@click.option(
    "--deps",
    default=None,
    help="Comma-separated subset of configured dependencies to probe.",
)
@click.option("--json", "as_json", is_flag=True, help="Emit the report as JSON.")
@click.pass_context
def health(ctx: click.Context, live: bool, deps: str | None, as_json: bool) -> None:
    """US-6.6: Run liveness/readiness checks (usable manually and in CI).

    Exits non-zero when the service is UNHEALTHY, so it can gate CI and
    deployment pipelines.
    """
    cfg = ctx.obj["CONFIG"]
    only = {d.strip() for d in deps.split(",")} if deps else None
    checker = _build_checker_from_config(cfg, only_deps=only)

    report = checker.check_liveness() if live else checker.check_readiness()

    if as_json:
        console.print_json(data=report.to_dict())
    else:
        colour = {
            HealthStatus.HEALTHY: "green",
            HealthStatus.DEGRADED: "yellow",
            HealthStatus.UNHEALTHY: "red",
        }[report.status]
        console.print(f"[bold {colour}]{report.service}: {report.status.value.upper()}[/]")
        for check in report.checks:
            mark = "✔" if check.status == HealthStatus.HEALTHY else "✘"
            tag = "" if check.critical else " (non-critical)"
            latency = f" [{check.latency_ms}ms]" if check.latency_ms is not None else ""
            console.print(f"  {mark} {check.name}{tag}: {check.detail}{latency}")

    # Non-zero exit only on a hard failure so degraded still passes CI gates
    # that merely require the process to be serviceable.
    if report.status == HealthStatus.UNHEALTHY:
        sys.exit(1)


# ── US-6.5 Deployment ─────────────────────────────────────────────────────────

ENVIRONMENTS_DIR = Path("configs/environments")


@cli.command(name="environments")
def environments() -> None:
    """US-6.5: List the deployment environments that are configured."""
    envs = list_environments(ENVIRONMENTS_DIR)
    if not envs:
        console.print(f"[yellow]No environment configs in {ENVIRONMENTS_DIR}/[/]")
        return
    console.print("[bold]Available environments:[/]")
    for name in envs:
        console.print(f"  • [cyan]{name}[/]")


@cli.command(name="deploy")
@click.option("--env", "environment", required=True, help="Environment to deploy.")
@click.option(
    "--build/--no-build",
    "build",
    default=None,
    help="Override the environment's compose build setting.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Show the deployment plan (resolved command + model versions) without running it.",
)
@click.option(
    "--skip-health",
    is_flag=True,
    help="Skip the post-deploy readiness gate.",
)
@click.option("--json", "as_json", is_flag=True, help="Emit the result as JSON.")
@click.pass_context
def deploy(
    ctx: click.Context,
    environment: str,
    build: bool | None,
    dry_run: bool,
    skip_health: bool,
    as_json: bool,
) -> None:
    """US-6.5: Run an automated, repeatable deployment for an environment.

    Resolves which model versions to ship from the registry (US-6.4), brings
    the services up via docker compose, then gates on readiness (US-6.6).
    Exits non-zero if the deployment fails.
    """
    deployer = Deployer(
        config_path=ctx.obj["CONFIG_PATH"],
        environments_dir=ENVIRONMENTS_DIR,
    )
    result = deployer.deploy(
        environment,
        build=build,
        dry_run=dry_run,
        skip_health=skip_health,
    )

    if as_json:
        console.print_json(data=result.to_dict())
    else:
        colour = {
            DeploymentStatus.SUCCESS: "green",
            DeploymentStatus.PLANNED: "cyan",
            DeploymentStatus.FAILED: "red",
        }[result.status]
        console.print(f"[bold {colour}]{result.environment}: {result.status.value.upper()}[/]")
        for step in result.steps:
            mark = "✔" if step.ok else "✘"
            console.print(f"  {mark} {step.name}: {step.detail}")
        if result.model_versions:
            console.print(f"  models: {result.model_versions}")
        if result.command:
            console.print(f"  command: [dim]{' '.join(result.command)}[/]")
        if result.message:
            console.print(result.message)

    if result.status == DeploymentStatus.FAILED:
        sys.exit(1)


if __name__ == "__main__":
    cli(obj={})

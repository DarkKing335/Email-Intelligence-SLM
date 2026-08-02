"""CLI entrypoint for Observability operations (Epic 5).

Commands for the Audit trail (US-5.3) and Monitoring alert checks (US-5.2):

    email-obs audit record --actor USER --action approve --resource EMAIL_ID
    email-obs audit query [--actor USER] [--resource EMAIL_ID] [--action ACTION]
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import click
import yaml
from rich.console import Console
from rich.table import Table

from email_observability.audit import AuditAction, AuditTrail

console = Console()

DEFAULT_CONFIG = Path("configs/observability.yaml")


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
    help="Path to the observability YAML configuration file.",
)
@click.pass_context
def cli(ctx: click.Context, config: Path) -> None:
    """Email Intelligence SLM — Observability operations (Epic 5)."""
    ctx.ensure_object(dict)
    ctx.obj["CONFIG"] = _load_config(config)


def _audit_trail(ctx: click.Context) -> AuditTrail:
    audit_cfg = ctx.obj["CONFIG"].get("audit", {})
    return AuditTrail(path=audit_cfg.get("path", "reports/audit/audit.jsonl"))


# ── US-5.3 Audit trail ────────────────────────────────────────────────────────


@cli.group()
def audit() -> None:
    """Record and review audit events."""


@audit.command(name="record")
@click.option("--actor", required=True, help="Who performed the action.")
@click.option(
    "--action",
    required=True,
    type=click.Choice([a.value for a in AuditAction]),
    help="Action type.",
)
@click.option("--resource", default=None, help="What was acted on (e.g. email id).")
@click.option("--outcome", default=None, help="Result of the action.")
@click.option("--model-version", default=None, help="Model version in effect.")
@click.option("--dataset-version", default=None, help="Dataset version in effect.")
@click.option("--detail", default=None, help="Free-text note.")
@click.pass_context
def audit_record(
    ctx: click.Context,
    actor: str,
    action: str,
    resource: str | None,
    outcome: str | None,
    model_version: str | None,
    dataset_version: str | None,
    detail: str | None,
) -> None:
    """US-5.3: Append an audit event."""
    trail = _audit_trail(ctx)
    event = trail.record(
        action,
        actor,
        resource=resource,
        outcome=outcome,
        model_version=model_version,
        dataset_version=dataset_version,
        detail=detail,
    )
    console.print(
        f"[bold green]✔ Recorded[/] {event.action.value} by {event.actor} ({event.id[:8]})"
    )


@audit.command(name="query")
@click.option("--actor", default=None, help="Filter by actor (user / email).")
@click.option("--resource", default=None, help="Filter by resource (e.g. email id).")
@click.option(
    "--action",
    default=None,
    type=click.Choice([a.value for a in AuditAction]),
    help="Filter by action type.",
)
@click.option("--since", default=None, help="ISO-8601 lower time bound.")
@click.option("--until", default=None, help="ISO-8601 upper time bound.")
@click.option("--limit", default=None, type=int, help="Return at most N most-recent events.")
@click.option("--json", "as_json", is_flag=True, help="Emit events as JSON.")
@click.pass_context
def audit_query(
    ctx: click.Context,
    actor: str | None,
    resource: str | None,
    action: str | None,
    since: str | None,
    until: str | None,
    limit: int | None,
    as_json: bool,
) -> None:
    """US-5.3: Review audit events, filterable by user or email."""
    trail = _audit_trail(ctx)
    events = trail.query(
        actor=actor,
        resource=resource,
        action=action,
        since=since,
        until=until,
        limit=limit,
    )
    if as_json:
        console.print_json(data=[e.model_dump(mode="json") for e in events])
        return
    if not events:
        console.print("[yellow]No matching audit events.[/]")
        return

    table = Table(title="Audit Trail")
    table.add_column("Time", style="dim")
    table.add_column("Actor", style="cyan")
    table.add_column("Action", style="magenta")
    table.add_column("Resource", style="green")
    table.add_column("Outcome", style="green")
    table.add_column("Model", style="blue")
    for e in events:
        table.add_row(
            e.timestamp.replace("T", " ")[:19],
            e.actor,
            e.action.value,
            e.resource or "—",
            e.outcome or "—",
            e.model_version or "—",
        )
    console.print(table)


if __name__ == "__main__":
    cli(obj={})

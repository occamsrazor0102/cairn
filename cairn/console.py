"""`beet` console script — thin dispatch over the Python module surface.

Exposes three shell subcommands, each backed by a Python entry point that
exists today:

  cairn gui      -> beet.gui.server.serve (stdlib HTTP, static SPA)
  cairn sidecar  -> beet.sidecar.main (stdio JSON-RPC, used by Tauri)
  cairn serve    -> uvicorn + beet.api.create_app (requires `cairn[api]`)

Analyze / eval / ablation subcommands from the pre-refactor CLI are not
restored here; use `cairn gui` (web UI) or the `/evaluation/run` /
`/analyze` endpoints directly until a replacement lands.
"""
from __future__ import annotations

import sys
from pathlib import Path

import click

from cairn.config import load_config, CONFIGS_DIR
from cairn.pipeline import CairnPipeline

_PROFILE_CHOICES = sorted(p.stem for p in CONFIGS_DIR.glob("*.yaml") if not p.name.startswith("_"))


@click.group()
def main() -> None:
    """CAIRN — LLM authorship detection."""


@main.command("gui")
@click.option("--host", default="127.0.0.1", show_default=True)
@click.option("--port", default=8877, show_default=True, type=int)
@click.option("--profile", "-p", default="default",
              type=click.Choice(_PROFILE_CHOICES), show_default=True,
              help="Config profile from configs/<name>.yaml.")
@click.option("--config", "-c", default=None, type=click.Path(exists=True),
              help="Path to a config YAML (overrides --profile).")
@click.option("--no-browser", is_flag=True, help="Don't auto-open the browser.")
def gui(host: str, port: int, profile: str, config: str | None, no_browser: bool) -> None:
    """Run the local web UI (stdlib HTTP, static SPA at /)."""
    from cairn.gui.server import serve

    config_path = Path(config) if config else (Path(__file__).parent.parent / "configs" / f"{profile}.yaml")
    cfg = load_config(config_path)
    pipeline = CairnPipeline(cfg)
    serve(
        pipeline,
        host=host, port=port, open_browser=not no_browser,
        config=cfg, profile=profile,
    )


@main.command("sidecar")
@click.option("--profile", "-p", default="default",
              type=click.Choice(_PROFILE_CHOICES), show_default=True)
@click.option("--config", "-c", default=None, type=click.Path(exists=True))
@click.option("--feedback-path", default="data/reviewer_feedback.jsonl", show_default=True)
@click.option("--log-level", default="WARNING",
              type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"]), show_default=True)
def sidecar_cmd(profile: str, config: str | None, feedback_path: str, log_level: str) -> None:
    """Run the stdio JSON-RPC sidecar (consumed by the Tauri shell)."""
    from cairn.sidecar import main as sidecar_main

    argv = [
        "--profile", profile,
        "--feedback-path", feedback_path,
        "--log-level", log_level,
    ]
    if config:
        argv += ["--config", config]
    sys.exit(sidecar_main(argv))


@main.command("serve")
@click.option("--host", default="127.0.0.1", show_default=True)
@click.option("--port", default=8000, show_default=True, type=int)
@click.option("--profile", "-p", default="default",
              type=click.Choice(_PROFILE_CHOICES), show_default=True)
@click.option("--config", "-c", default=None, type=click.Path(exists=True))
def serve_cmd(host: str, port: int, profile: str, config: str | None) -> None:
    """Run the FastAPI HTTP API (requires `pip install cairn[api]`)."""
    try:
        import uvicorn
    except ImportError:
        raise click.ClickException(
            "`cairn serve` requires the [api] extra: `pip install cairn[api]`"
        )
    from cairn.api import create_app

    config_path = Path(config) if config else (Path(__file__).parent.parent / "configs" / f"{profile}.yaml")
    app = create_app(config_path)
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()

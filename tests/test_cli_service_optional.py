"""The command-line interface must work without the optional service extra.

These tests do not need FastAPI. Where they simulate its absence, they block the import explicitly,
so the result does not depend on what happens to be installed.
"""

import subprocess
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from microflow.cli import app
from microflow.service.settings import TOKEN_ENV

runner = CliRunner()
ROOT = Path(__file__).resolve().parent.parent
TOKEN = "s" * 40


def _settings_file(tmp_path: Path) -> Path:
    path = tmp_path / "service.yaml"
    path.write_text(
        f"state_directory: {tmp_path / 'state'}\n"
        f"input_roots: [{tmp_path}]\n"
        f"output_roots: [{tmp_path}]\n",
        encoding="utf-8",
    )
    return path


def test_importing_the_cli_does_not_load_the_web_stack() -> None:
    code = (
        "import sys, microflow.cli; "
        "print(int('fastapi' in sys.modules), int('uvicorn' in sys.modules))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, cwd=ROOT, check=True
    )
    assert result.stdout.strip() == "0 0"


def test_the_pipeline_commands_run_with_the_web_stack_blocked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "fastapi", None)
    monkeypatch.setitem(sys.modules, "uvicorn", None)
    result = runner.invoke(app, ["validate-config", str(ROOT / "examples" / "synthetic.yaml")])
    assert result.exit_code == 0, result.output
    assert "valid:" in result.output


def test_serve_reports_how_to_install_the_extra_when_it_is_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = tmp_path / "service.yaml"
    settings.write_text(
        f"state_directory: {tmp_path / 'state'}\n"
        f"input_roots: [{tmp_path}]\n"
        f"output_roots: [{tmp_path}]\n",
        encoding="utf-8",
    )
    monkeypatch.setenv(TOKEN_ENV, TOKEN)
    monkeypatch.setitem(sys.modules, "fastapi", None)
    result = runner.invoke(app, ["serve", str(settings)])
    assert result.exit_code == 1
    assert "uv sync --extra service" in result.output
    assert TOKEN not in result.output


def test_serve_refuses_to_start_without_a_token_and_never_prints_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _settings_file(tmp_path)
    monkeypatch.delenv(TOKEN_ENV, raising=False)
    result = runner.invoke(app, ["serve", str(settings)])
    assert result.exit_code == 1
    assert TOKEN_ENV in result.output


def test_serve_refuses_a_remote_address_without_the_explicit_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _settings_file(tmp_path)
    monkeypatch.setenv(TOKEN_ENV, TOKEN)
    result = runner.invoke(app, ["serve", str(settings), "--host", "0.0.0.0"])
    assert result.exit_code == 1
    assert "refusing to listen on 0.0.0.0" in result.output


def test_serve_reports_a_missing_settings_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(TOKEN_ENV, TOKEN)
    result = runner.invoke(app, ["serve", str(tmp_path / "absent.yaml")])
    assert result.exit_code != 0


def test_the_example_settings_file_loads_without_the_extra() -> None:
    from microflow.service.settings import load_settings

    settings = load_settings(ROOT / "examples" / "service" / "service.yaml")
    assert settings.host == "127.0.0.1"
    assert settings.allow_remote is False


def test_the_help_lists_the_service_command() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "serve" in result.output

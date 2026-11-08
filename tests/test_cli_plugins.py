import textwrap
from collections.abc import Callable
from pathlib import Path

import numpy as np
from typer.testing import CliRunner

from microflow.cli import app
from microflow.image_io import write_image
from microflow.plugins.api import ENTRY_POINT_GROUPS

runner = CliRunner()

MODULE = textwrap.dedent(
    """
    import numpy as np
    from pydantic import BaseModel, ConfigDict
    from microflow.plugins.api import PLUGIN_API_VERSION, FeatureColumn


    class NoOptions(BaseModel):
        model_config = ConfigDict(extra="forbid")


    class Feature:
        name = "cli-feature"
        version = "3.1"
        api_version = PLUGIN_API_VERSION
        kind = "feature"
        options_model = NoOptions

        def features(self):
            return (FeatureColumn("constant", "1", "always one"),)

        def measure(self, labels, plane, calibration, options):
            return {int(i): {"constant": 1.0} for i in np.unique(labels) if i > 0}
    """
)


def _install(plugin_site: Callable[..., None]) -> None:
    plugin_site(
        "cli-dist",
        {ENTRY_POINT_GROUPS["feature"]: {"cli-feature": "mf_cli:Feature"}},
        {"mf_cli.py": MODULE},
    )


def _config(root: Path, plugins: str = "") -> Path:
    (root / "data").mkdir(parents=True, exist_ok=True)
    write_image(root / "data" / "img.tif", np.full((8, 8), 100, dtype=np.uint16))
    text = textwrap.dedent(
        """
        name: cli-plugins
        dataset:
          images:
            - name: img
              path: data/img.tif
              calibration: {pixel_size: 1.0, unit: um}
        output_directory: out
        workers: 1
        """
    )
    path = root / "batch.yaml"
    path.write_text(text + plugins, encoding="utf-8")
    return path


def test_plugins_list_shows_installed_plugins_without_running_them(
    plugin_site: Callable[..., None],
) -> None:
    _install(plugin_site)
    result = runner.invoke(app, ["plugins", "list"])
    assert result.exit_code == 0, result.output
    assert "cli-feature" in result.output
    assert "cli-dist" in result.output
    assert "mf_cli:Feature" in result.output


def test_plugins_check_reports_each_named_plugin_with_its_version(
    plugin_site: Callable[..., None], tmp_path: Path
) -> None:
    _install(plugin_site)
    config = _config(tmp_path, "features:\n  - name: cli-feature\n")
    result = runner.invoke(app, ["plugins", "check", str(config)])
    assert result.exit_code == 0, result.output
    assert "ok   feature      cli-feature version 3.1" in result.output


def test_plugins_check_fails_for_a_plugin_that_is_not_installed(tmp_path: Path) -> None:
    config = _config(tmp_path, "features:\n  - name: absent-feature\n")
    result = runner.invoke(app, ["plugins", "check", str(config)])
    assert result.exit_code == 1
    failure = "FAIL feature      absent-feature: no feature plugin named 'absent-feature'"
    assert failure in result.output


def test_plugins_check_says_when_a_configuration_names_no_plugins(tmp_path: Path) -> None:
    result = runner.invoke(app, ["plugins", "check", str(_config(tmp_path))])
    assert result.exit_code == 0
    assert "no plugins are named in this configuration" in result.output


def test_batch_refuses_a_missing_plugin_with_exit_code_one_and_writes_nothing(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path, "features:\n  - name: absent-feature\n")
    result = runner.invoke(app, ["batch", str(config)])
    assert result.exit_code == 1
    assert "error: no feature plugin named 'absent-feature' is installed" in result.output
    assert not (tmp_path / "out").exists()


def test_batch_runs_an_installed_plugin_and_reports_its_status(
    plugin_site: Callable[..., None], tmp_path: Path
) -> None:
    _install(plugin_site)
    config = _config(tmp_path, "features:\n  - name: cli-feature\n")
    result = runner.invoke(app, ["batch", str(config)])
    assert result.exit_code == 0, result.output
    assert "status: complete" in result.output

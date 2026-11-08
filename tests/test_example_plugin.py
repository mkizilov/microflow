import importlib
import os
import shutil
import subprocess
import sys
import tomllib
from collections.abc import Callable, Iterator
from pathlib import Path

import numpy as np
import pytest

from microflow.batch import BatchConfig, run_batch
from microflow.config import SegmentationConfig
from microflow.image_io import write_image
from microflow.loading import load_array
from microflow.plugins.api import ENTRY_POINT_GROUPS, KINDS, PluginError, PluginSpec
from microflow.plugins.contracts import check_features, check_findings
from microflow.plugins.registry import configure, discover, load
from microflow.preprocessing import preprocess
from microflow.preprocessing.steps import PluginStep
from microflow.segmentation import segment_plane
from microflow.units import Area, Calibration

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "plugins" / "microflow-example-plugins"
DIST = "microflow-example-plugins"
EXPECTED_NAMES = {
    "preprocessor": "example-offset",
    "segmenter": "example-quantile",
    "feature": "example-object-range",
    "qc": "example-small-fraction",
}


def _declared_entry_points() -> dict[str, dict[str, str]]:
    data = tomllib.loads((EXAMPLE / "pyproject.toml").read_text(encoding="utf-8"))
    return data["project"]["entry-points"]


def _purge_modules_under(root: Path) -> None:
    for name, module in list(sys.modules.items()):
        if str(getattr(module, "__file__", "") or "").startswith(str(root)):
            del sys.modules[name]


@pytest.fixture
def example_on_path(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    source = EXAMPLE / "src"
    monkeypatch.syspath_prepend(str(source))
    yield
    _purge_modules_under(source)


@pytest.fixture
def example_installed(plugin_site: Callable[..., None], example_on_path: None) -> None:
    plugin_site(DIST, _declared_entry_points())


# --- the example's declaration --------------------------------------------------------------------


def test_example_declares_one_plugin_per_kind_under_the_documented_groups() -> None:
    declared = _declared_entry_points()
    assert set(declared) == set(ENTRY_POINT_GROUPS.values())
    assert {group: set(entries) for group, entries in declared.items()} == {
        ENTRY_POINT_GROUPS[kind]: {EXPECTED_NAMES[kind]} for kind in KINDS
    }


def test_example_targets_import_and_match_their_entry_point_names(
    example_on_path: None,
) -> None:
    kinds_by_group = {group: kind for kind, group in ENTRY_POINT_GROUPS.items()}
    for group, entries in _declared_entry_points().items():
        for name, target in entries.items():
            module_name, attribute = target.split(":")
            plugin = getattr(importlib.import_module(module_name), attribute)
            assert plugin.name == name
            assert plugin.kind == kinds_by_group[group]
            assert plugin.api_version == 1


# --- installation and registration ----------------------------------------------------------------


def test_example_is_discovered_and_passes_load_validation_when_registered(
    example_installed: None,
) -> None:
    found = {item.name: item.distribution for item in discover()}
    for name in EXPECTED_NAMES.values():
        assert found[name] == DIST
    for kind, name in EXPECTED_NAMES.items():
        assert load(kind, name).version == "1.0.0"


@pytest.mark.skipif(
    shutil.which("uv") is None,
    reason="installing the example needs uv, which is not available in every environment",
)
def test_example_installs_with_uv_into_a_private_target_and_is_discovered(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "target"
    env = {**os.environ, "UV_CACHE_DIR": str(tmp_path / "uv-cache")}
    subprocess.run(
        ["uv", "pip", "install", "--no-deps", "--target", str(target), str(EXAMPLE)],
        check=True,
        env=env,
        timeout=600,
    )
    monkeypatch.syspath_prepend(str(target))
    try:
        found = {item.name: item.distribution for item in discover()}
        for name in EXPECTED_NAMES.values():
            assert found[name] == DIST
        for kind, name in EXPECTED_NAMES.items():
            assert load(kind, name).name == name
    finally:
        _purge_modules_under(target)


# --- behaviour through the pipelines --------------------------------------------------------------


def test_example_segmenter_separates_objects_above_the_chosen_quantile(
    example_installed: None,
) -> None:
    plane = np.zeros((20, 20))
    plane[2:6, 2:6] = 10.0
    plane[12:14, 12:14] = 10.0
    config = SegmentationConfig(
        method="plugin",
        plugin=PluginSpec(name="example-quantile"),
        min_object_area=Area(value=1.0, unit="um"),
    )
    labels, _ = segment_plane(plane, config, Calibration(pixel_size=1.0, unit="um"))
    areas = sorted(int((labels == value).sum()) for value in range(1, int(labels.max()) + 1))
    assert areas == [4, 16]


def test_example_segmenter_returns_no_objects_for_an_empty_plane(example_installed: None) -> None:
    config = SegmentationConfig(
        method="plugin",
        plugin=PluginSpec(name="example-quantile"),
        min_object_area=Area(value=1.0, unit="um"),
    )
    labels, _ = segment_plane(
        np.full((5, 5), np.nan), config, Calibration(pixel_size=1.0, unit="um")
    )
    assert int(labels.max()) == 0


def test_example_segmenter_rejects_an_out_of_range_quantile(example_installed: None) -> None:
    with pytest.raises(PluginError, match="invalid options: quantile"):
        configure("segmenter", PluginSpec(name="example-quantile", options={"quantile": 1.5}))


def test_example_preprocessor_adds_its_offset_and_keeps_invalid_pixels(
    example_installed: None,
) -> None:
    image = load_array(np.array([[1.0, 2.0], [3.0, np.nan]]), axes="YX")
    step = PluginStep.model_validate(
        {"kind": "plugin", "plugin": "example-offset", "options": {"offset": 2.5}}
    )
    result = preprocess(image, [step])
    np.testing.assert_allclose(result.data[0], [3.5, 4.5])
    np.testing.assert_allclose(result.data[1, 0], 5.5)
    assert np.isnan(result.data[1, 1])


def test_example_feature_reports_the_intensity_range_of_each_object(
    example_installed: None,
) -> None:
    configured = configure("feature", PluginSpec(name="example-object-range"))
    labels = np.zeros((4, 4), dtype=np.int32)
    labels[0:2, 0:2] = 1
    labels[3, 3] = 2
    plane = np.arange(16, dtype=np.float64).reshape(4, 4)
    plane[0, 0] = np.nan

    values = configured.plugin.measure(labels, plane, None, configured.options)
    table = check_features(values, {1, 2}, configured.plugin.features(), "example")
    assert table[1]["intensity_range"] == pytest.approx(4.0)
    assert table[2]["intensity_range"] == 0.0


def test_example_quality_check_warns_only_when_small_objects_dominate(
    example_installed: None,
) -> None:
    configured = configure(
        "qc", PluginSpec(name="example-small-fraction", options={"min_pixels": 4})
    )
    plane = np.zeros((6, 6))
    crowded = np.zeros((6, 6), dtype=np.int32)
    crowded[0, 0] = 1
    crowded[0, 2] = 2
    crowded[3:6, 3:6] = 3
    findings = check_findings(
        configured.plugin.check(crowded, plane, None, configured.options), "qc"
    )
    assert len(findings) == 1
    assert "67% of objects are smaller than 4 pixels" in findings[0]

    sparse = np.zeros((6, 6), dtype=np.int32)
    sparse[1:5, 1:5] = 1
    sparse_findings = configured.plugin.check(sparse, plane, None, configured.options)
    assert check_findings(sparse_findings, "qc") == ()


def test_example_plugins_run_through_a_batch(
    example_installed: None, tmp_path: Path
) -> None:
    pixels = np.full((40, 40), 100, dtype=np.uint16)
    pixels[5:11, 5:11] = 2000
    pixels[25:29, 25:29] = 3000
    path = tmp_path / "data" / "img.tif"
    path.parent.mkdir(parents=True)
    write_image(path, pixels)
    config = BatchConfig.model_validate(
        {
            "name": "example-batch",
            "dataset": {
                "images": [
                    {"name": "img", "path": path, "calibration": {"pixel_size": 1.0, "unit": "um"}}
                ]
            },
            "segmentation": {"min_object_area": {"value": 4, "unit": "um"}},
            "features": [{"name": "example-object-range"}],
            "quality_checks": [{"name": "example-small-fraction"}],
            "output_directory": tmp_path / "out",
            "workers": 1,
        }
    )
    manifest = run_batch(config)
    assert manifest.status == "complete"
    assert "feature:example-object-range" in manifest.items[0].outputs
    assert manifest.provenance.algorithm_versions["plugin:qc:example-small-fraction"] == "1.0.0"

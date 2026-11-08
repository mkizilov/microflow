import csv
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from pydantic import ValidationError

from microflow.batch import BatchConfig, analysis_id, run_batch
from microflow.batch.engine import _analysis_settings
from microflow.batch.store import item_directory_name
from microflow.config import SegmentationConfig
from microflow.image_io import write_image
from microflow.loading import load_array
from microflow.plugins.api import ENTRY_POINT_GROUPS, PluginError, PluginSpec
from microflow.preprocessing import preprocess
from microflow.preprocessing.steps import PluginStep
from microflow.segmentation import segment_plane
from microflow.units import Area, Calibration

MODULE = "mf_integration"

INTEGRATION_MODULE = '''
import numpy as np
from pydantic import BaseModel, ConfigDict

from microflow.plugins.api import PLUGIN_API_VERSION, FeatureColumn


class NoOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Level(BaseModel):
    model_config = ConfigDict(extra="forbid")
    level: float = 0.5


class _Base:
    version = "1.0"
    api_version = PLUGIN_API_VERSION


class GoodSegmenter(_Base):
    name = "good-seg"
    kind = "segmenter"
    options_model = Level

    def segment(self, plane, calibration, options):
        return (plane > options.level).astype(np.int32)


class WrongShapeSegmenter(_Base):
    name = "wrong-shape-seg"
    kind = "segmenter"
    options_model = NoOptions

    def segment(self, plane, calibration, options):
        return np.zeros((2, 2), dtype=np.int32)


class InPlacePreprocessor(_Base):
    name = "in-place"
    kind = "preprocessor"
    options_model = NoOptions

    def process(self, data, calibration, options):
        data += 1000.0
        return data


class InfPreprocessor(_Base):
    name = "inf-pre"
    kind = "preprocessor"
    options_model = NoOptions

    def process(self, data, calibration, options):
        out = data.copy()
        out[0, 0] = np.inf
        return out


class PixelCountFeature(_Base):
    name = "good-feature"
    kind = "feature"
    options_model = NoOptions

    def features(self):
        return (FeatureColumn("pixel_count", "px", "number of pixels in the object"),)

    def measure(self, labels, plane, calibration, options):
        ids, counts = np.unique(labels[labels > 0], return_counts=True)
        return {int(i): {"pixel_count": float(c)} for i, c in zip(ids, counts)}


class SaturationFeature(_Base):
    name = "bad-feature"
    kind = "feature"
    options_model = NoOptions

    def features(self):
        return (FeatureColumn("mean_value", "intensity", "mean intensity of the object"),)

    def measure(self, labels, plane, calibration, options):
        if np.nanmax(plane) >= 60000:
            raise RuntimeError("saturated plane")
        ids = [int(i) for i in np.unique(labels) if i > 0]
        return {i: {"mean_value": float(plane[labels == i].mean())} for i in ids}


class MutatingFeature(_Base):
    name = "mutating-feature"
    kind = "feature"
    options_model = NoOptions

    def features(self):
        return (FeatureColumn("constant", "1", "always one"),)

    def measure(self, labels, plane, calibration, options):
        objects = [int(value) for value in np.unique(labels) if value]
        plane[...] = -1.0
        labels[...] = 0
        return {label: {"constant": 1.0} for label in objects}


class WarningQc(_Base):
    name = "always-warn"
    kind = "qc"
    options_model = NoOptions

    def check(self, labels, plane, calibration, options):
        return ["check ran"]
'''


@pytest.fixture
def installed(plugin_site: Callable[..., None]) -> None:
    plugin_site(
        "integration-dist",
        {
            ENTRY_POINT_GROUPS["segmenter"]: {
                "good-seg": f"{MODULE}:GoodSegmenter",
                "wrong-shape-seg": f"{MODULE}:WrongShapeSegmenter",
            },
            ENTRY_POINT_GROUPS["preprocessor"]: {
                "in-place": f"{MODULE}:InPlacePreprocessor",
                "inf-pre": f"{MODULE}:InfPreprocessor",
            },
            ENTRY_POINT_GROUPS["feature"]: {
                "good-feature": f"{MODULE}:PixelCountFeature",
                "bad-feature": f"{MODULE}:SaturationFeature",
                "mutating-feature": f"{MODULE}:MutatingFeature",
            },
            ENTRY_POINT_GROUPS["qc"]: {"always-warn": f"{MODULE}:WarningQc"},
        },
        {f"{MODULE}.py": INTEGRATION_MODULE},
    )


CALIBRATION = Calibration(pixel_size=1.0, unit="um")


def _image(root: Path, name: str, *, saturate: bool) -> dict[str, Any]:
    """Two bright blocks on a dim background. ``saturate`` adds one pixel at 60000."""
    pixels = np.full((40, 40), 100, dtype=np.uint16)
    pixels[5:11, 5:11] = 2000
    pixels[25:29, 25:29] = 3000
    if saturate:
        pixels[0, 0] = 60000
    path = root / "data" / f"{name}.tif"
    path.parent.mkdir(parents=True, exist_ok=True)
    write_image(path, pixels)
    return {"name": name, "path": path, "calibration": {"pixel_size": 1.0, "unit": "um"}}


def _config(root: Path, images: list[dict[str, Any]], **overrides: Any) -> BatchConfig:
    values: dict[str, Any] = {
        "name": "plugin-batch",
        "dataset": {"images": images},
        "segmentation": {"min_object_area": {"value": 4, "unit": "um"}},
        "output_directory": root / "out",
        "workers": 1,
    }
    values.update(overrides)
    return BatchConfig.model_validate(values)


def _item_dir(config: BatchConfig, name: str) -> Path:
    return config.output_directory / "items" / item_directory_name(name)


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


# --- segmentation -------------------------------------------------------------------------------


def test_segmenter_output_goes_through_the_standard_size_rules(installed: None) -> None:
    plane = np.zeros((6, 6))
    plane[1:3, 1:3] = 1.0
    plane[4, 4] = 1.0
    keep = SegmentationConfig(
        method="plugin",
        plugin=PluginSpec(name="good-seg", options={"level": 0.5}),
        min_object_area=Area(value=2.0, unit="um"),
    )
    labels, level = segment_plane(plane, keep, CALIBRATION)
    assert level is None
    assert int(labels.max()) == 1
    assert int((labels > 0).sum()) == 5

    drop = keep.model_copy(update={"min_object_area": Area(value=6.0, unit="um")})
    removed, _ = segment_plane(plane, drop, CALIBRATION)
    assert int((removed > 0).sum()) == 0


def test_segmenter_output_that_breaks_its_contract_fails_with_the_plugin_named(
    installed: None,
) -> None:
    config = SegmentationConfig(
        method="plugin",
        plugin=PluginSpec(name="wrong-shape-seg"),
        min_object_area=Area(value=1.0, unit="um"),
    )
    with pytest.raises(PluginError, match=r"segmenter plugin 'wrong-shape-seg'.*returned shape"):
        segment_plane(np.zeros((6, 6)), config, CALIBRATION)


# --- preprocessing ------------------------------------------------------------------------------


def test_preprocessor_plugin_cannot_alter_the_input_image(installed: None) -> None:
    image = load_array(np.arange(16, dtype=np.uint16).reshape(4, 4), axes="YX")
    original = image.data.copy()
    step = PluginStep.model_validate({"kind": "plugin", "plugin": "in-place"})

    result = preprocess(image, [step])

    np.testing.assert_array_equal(image.data, original)
    np.testing.assert_allclose(result.data, original.astype(np.float64) + 1000.0)
    record = next(item for item in result.records if item.stage == "plugin:in-place")
    assert record.parameters["version"] == "1.0"


def test_preprocessor_output_with_infinite_values_is_refused(installed: None) -> None:
    image = load_array(np.ones((4, 4), dtype=np.float64), axes="YX")
    step = PluginStep.model_validate({"kind": "plugin", "plugin": "inf-pre"})
    with pytest.raises(PluginError, match="preprocessor plugin 'inf-pre'.*infinite values"):
        preprocess(image, [step])


# --- batch runs ---------------------------------------------------------------------------------


def test_feature_plugin_writes_a_table_that_agrees_with_the_built_in_areas(
    installed: None, tmp_path: Path
) -> None:
    config = _config(
        tmp_path,
        [_image(tmp_path, "img", saturate=False)],
        features=[{"name": "good-feature"}],
    )
    manifest = run_batch(config)

    assert manifest.status == "complete"
    outputs = manifest.items[0].outputs
    assert {"feature:good-feature", "feature_dictionary:good-feature"} <= set(outputs)
    assert manifest.provenance.algorithm_versions["plugin:feature:good-feature"] == "1.0"

    item_dir = _item_dir(config, "img")
    features = {
        (row["plane"], row["label"]): float(row["pixel_count"])
        for row in _read_rows(item_dir / "features_good-feature.csv")
    }
    areas = {
        (row["plane"], row["label"]): float(row["area_um2"])
        for row in _read_rows(item_dir / "measurements.csv")
    }
    assert features and features == areas

    dictionary_rows = _read_rows(item_dir / "features_good-feature_dictionary.csv")
    dictionary = {row["name"]: row for row in dictionary_rows}
    assert dictionary["pixel_count"]["unit"] == "px"


def test_failing_feature_plugin_fails_only_its_image_and_publishes_nothing_for_it(
    installed: None, tmp_path: Path
) -> None:
    images = [_image(tmp_path, "good", saturate=False), _image(tmp_path, "bad", saturate=True)]
    config = _config(tmp_path, images, features=[{"name": "bad-feature"}])
    manifest = run_batch(config)

    items = {item.name: item for item in manifest.items}
    assert manifest.status == "completed_with_failures"
    assert items["good"].status == "ok"
    assert items["bad"].status == "failed"
    assert (
        "feature plugin 'bad-feature' failed during measure: RuntimeError: saturated plane"
        in items["bad"].error
    )

    bad_dir = _item_dir(config, "bad")
    assert not (bad_dir / "labels.tif").exists()
    assert not (bad_dir / "measurements.csv").exists()
    assert not list(bad_dir.glob("features_*"))
    assert (_item_dir(config, "good") / "features_bad-feature.csv").exists()


def test_feature_plugin_cannot_change_the_built_in_measurements(
    installed: None, tmp_path: Path
) -> None:
    images = [_image(tmp_path, "img", saturate=False)]
    plain = run_batch(_config(tmp_path / "plain", images))
    mutated = run_batch(
        _config(tmp_path / "mutated", images, features=[{"name": "mutating-feature"}])
    )

    assert plain.status == mutated.status == "complete"
    plain_csv = _item_dir(_config(tmp_path / "plain", images), "img") / "measurements.csv"
    mutated_csv = _item_dir(_config(tmp_path / "mutated", images), "img") / "measurements.csv"
    assert plain_csv.read_bytes() == mutated_csv.read_bytes()

    feature_rows = _read_rows(
        _item_dir(_config(tmp_path / "mutated", images), "img") / "features_mutating-feature.csv"
    )
    assert feature_rows and {row["constant"] for row in feature_rows} == {"1.0"}


def test_quality_plugin_findings_are_warnings_and_do_not_fail_the_image(
    installed: None, tmp_path: Path
) -> None:
    config = _config(
        tmp_path,
        [_image(tmp_path, "img", saturate=False)],
        quality_checks=[{"name": "always-warn"}],
    )
    manifest = run_batch(config)

    assert manifest.status == "complete"
    assert any(
        text.startswith("qc plugin always-warn [") and text.endswith("check ran")
        for text in manifest.items[0].warnings
    )


def test_segmentation_and_preprocessing_plugins_run_in_a_batch(
    installed: None, tmp_path: Path
) -> None:
    config = _config(
        tmp_path,
        [_image(tmp_path, "img", saturate=False)],
        preprocessing=[{"kind": "plugin", "plugin": "in-place"}],
        segmentation={
            "method": "plugin",
            "plugin": {"name": "good-seg", "options": {"level": 1500}},
            "min_object_area": {"value": 4, "unit": "um"},
        },
    )
    manifest = run_batch(config)

    assert manifest.status == "complete"
    assert manifest.items[0].objects >= 1
    assert "plugin preprocessor ran" in " ".join(manifest.items[0].warnings)
    assert manifest.provenance.algorithm_versions["plugin:segmenter:good-seg"] == "1.0"


def test_naming_an_unavailable_plugin_refuses_the_batch_before_any_output(tmp_path: Path) -> None:
    config = _config(
        tmp_path, [_image(tmp_path, "img", saturate=False)], features=[{"name": "not-installed"}]
    )
    with pytest.raises(PluginError, match="no feature plugin named 'not-installed' is installed"):
        run_batch(config)
    assert not config.output_directory.exists()


# --- trust and configuration --------------------------------------------------------------------


def test_a_plugin_cannot_be_listed_twice_in_one_section(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="features lists good-feature more than once"):
        _config(
            tmp_path,
            [_image(tmp_path, "img", saturate=False)],
            features=[{"name": "good-feature"}, {"name": "good-feature"}],
        )


def test_datasets_cannot_name_a_plugin(tmp_path: Path) -> None:
    images = [{"name": "img", "path": tmp_path / "img.tif", "plugin": "good-seg"}]
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        _config(tmp_path, images)


def test_plugin_steps_reject_unknown_keys(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        PluginStep.model_validate({"kind": "plugin", "plugin": "in-place", "script": "x.py"})


def test_plugin_segmentation_needs_a_plugin_and_forbids_built_in_options() -> None:
    with pytest.raises(ValidationError, match="plugin is required when method is 'plugin'"):
        SegmentationConfig(method="plugin")
    with pytest.raises(ValidationError, match="plugin is only used when method is 'plugin'"):
        SegmentationConfig(method="otsu", plugin=PluginSpec(name="good-seg"))
    with pytest.raises(ValidationError, match="not used when method is 'plugin'"):
        SegmentationConfig(method="plugin", plugin=PluginSpec(name="good-seg"), watershed=True)
    with pytest.raises(ValidationError, match="adaptive_offset not used when method is 'plugin'"):
        SegmentationConfig(method="plugin", plugin=PluginSpec(name="good-seg"), adaptive_offset=2.0)
    SegmentationConfig(method="plugin", plugin=PluginSpec(name="good-seg"))


# --- identity -----------------------------------------------------------------------------------


def test_configurations_without_plugins_keep_their_settings_and_reuse_results(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path, [_image(tmp_path, "img", saturate=False)])
    settings = _analysis_settings(config)
    assert set(settings) == {
        "preprocessing",
        "segmentation",
        "segmentation_channel",
        "quality_control",
    }
    assert "plugin" not in settings["segmentation"]

    run_batch(config)
    assert run_batch(config).counts["resumed"] == 1


def test_plugin_options_and_versions_change_the_analysis_identity(
    installed: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    images = [_image(tmp_path, "img", saturate=False)]
    plain = _config(tmp_path, images)
    with_feature = _config(tmp_path, images, features=[{"name": "good-feature"}])
    assert analysis_id(with_feature) != analysis_id(plain)

    level_a = _config(
        tmp_path,
        images,
        segmentation={
            "method": "plugin",
            "plugin": {"name": "good-seg", "options": {"level": 0.5}},
            "min_object_area": {"value": 4, "unit": "um"},
        },
    )
    level_b = level_a.model_copy(
        update={
            "segmentation": level_a.segmentation.model_copy(
                update={"plugin": PluginSpec(name="good-seg", options={"level": 0.25})}
            )
        }
    )
    assert analysis_id(level_a) != analysis_id(level_b)

    before = analysis_id(with_feature)
    monkeypatch.setattr(sys.modules[MODULE].PixelCountFeature, "version", "2.0")
    assert analysis_id(with_feature) != before

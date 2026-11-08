import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from microflow.batch import load_batch_config
from microflow.service.errors import ServiceError
from microflow.service.settings import Layout, prepare_layout
from microflow.service.validation import batch_summary, plugin_uses, prepare_batch, prepare_stats
from service_fixtures import Tree, batch_mapping, make_tree, settings_for, write_batch_config
from stats_fixtures import design_mapping, write_design, write_dictionary, write_measurements


def _layout(tmp_path: Path) -> tuple[Tree, Layout]:
    tree = make_tree(tmp_path)
    return tree, prepare_layout(settings_for(tree))


def _refusal(call: Callable[[], object]) -> ServiceError:
    with pytest.raises(ServiceError) as caught:
        call()
    return caught.value


def test_a_valid_batch_configuration_is_accepted_with_resolved_locations(tmp_path: Path) -> None:
    tree, layout = _layout(tmp_path)
    path = write_batch_config(tree.inputs, batch_mapping(tree, output="run-1"))
    prepared = prepare_batch(str(path), layout)
    assert prepared.config_path == path.resolve()
    assert prepared.output_directory == (tree.outputs / "run-1").resolve()
    assert prepared.output_name == "output/run-1"
    assert prepared.warnings == ()
    assert batch_summary(prepared) == {
        "name": "service-test",
        "images": 2,
        "output": "output/run-1",
        "workers": 1,
    }


def test_workers_above_the_service_limit_are_reduced_and_reported(tmp_path: Path) -> None:
    tree, layout = _layout(tmp_path)
    path = write_batch_config(tree.inputs, batch_mapping(tree, workers=4))
    prepared = prepare_batch(str(path), layout)
    assert any("reduced from 4 to 1" in warning for warning in prepared.warnings)
    assert batch_summary(prepared)["workers"] == 1


def test_a_configuration_outside_the_input_roots_is_refused_without_echoing_its_path(
    tmp_path: Path,
) -> None:
    tree, layout = _layout(tmp_path)
    elsewhere = tree.root / "elsewhere"
    elsewhere.mkdir()
    path = write_batch_config(elsewhere, batch_mapping(tree))
    error = _refusal(lambda: prepare_batch(str(path), layout))
    assert error.code == "path_not_allowed"
    assert str(tree.root) not in error.message


def test_an_image_outside_the_input_roots_is_refused_and_named(tmp_path: Path) -> None:
    tree, layout = _layout(tmp_path)
    secret = tree.root / "secret.tif"
    secret.write_bytes(b"x")
    mapping = batch_mapping(tree)
    mapping["dataset"]["images"][0]["path"] = str(secret)
    path = write_batch_config(tree.inputs, mapping)
    error = _refusal(lambda: prepare_batch(str(path), layout))
    assert error.code == "path_not_allowed"
    assert error.message.startswith("image 'a'")
    assert str(tree.root) not in error.message


def test_an_image_that_is_a_link_out_of_the_roots_is_refused(tmp_path: Path) -> None:
    tree, layout = _layout(tmp_path)
    secret = tree.root / "secret.tif"
    secret.write_bytes(b"x")
    link = tree.inputs / "link.tif"
    os.symlink(secret, link)
    mapping = batch_mapping(tree)
    mapping["dataset"]["images"][0]["path"] = str(link)
    path = write_batch_config(tree.inputs, mapping)
    assert _refusal(lambda: prepare_batch(str(path), layout)).code == "path_not_allowed"


def test_a_missing_image_is_reported_by_its_name(tmp_path: Path) -> None:
    tree, layout = _layout(tmp_path)
    mapping = batch_mapping(tree)
    (tree.inputs / "data" / "b.tif").unlink()
    path = write_batch_config(tree.inputs, mapping)
    error = _refusal(lambda: prepare_batch(str(path), layout))
    assert error.code == "not_found"
    assert error.message.startswith("image 'b'")


def test_an_output_outside_the_output_roots_is_refused(tmp_path: Path) -> None:
    tree, layout = _layout(tmp_path)
    mapping = batch_mapping(tree)
    mapping["output_directory"] = str(tree.root / "elsewhere")
    path = write_batch_config(tree.inputs, mapping)
    assert _refusal(lambda: prepare_batch(str(path), layout)).code == "path_not_allowed"


def test_an_output_directory_that_is_the_root_itself_is_refused(tmp_path: Path) -> None:
    tree, layout = _layout(tmp_path)
    mapping = batch_mapping(tree)
    mapping["output_directory"] = str(tree.outputs)
    path = write_batch_config(tree.inputs, mapping)
    assert _refusal(lambda: prepare_batch(str(path), layout)).code == "path_not_allowed"


PLUGIN_CHANGES: list[tuple[str, Callable[[dict[str, Any]], None], str]] = [
    ("features", lambda m: m.update(features=[{"name": "shape"}]), "features"),
    ("quality checks", lambda m: m.update(quality_checks=[{"name": "check"}]), "quality_checks"),
    (
        "segmentation plugin",
        lambda m: m.update(segmentation={"method": "plugin", "plugin": {"name": "seg"}}),
        "segmentation method 'plugin'",
    ),
    (
        "preprocessing plugin",
        lambda m: m.update(preprocessing=[{"kind": "plugin", "plugin": "clean-up"}]),
        "a plugin preprocessing step",
    ),
]


@pytest.mark.parametrize(
    ("label", "change", "fragment"), PLUGIN_CHANGES, ids=[change[0] for change in PLUGIN_CHANGES]
)
def test_configurations_that_use_plugins_are_refused(
    tmp_path: Path, label: str, change: Callable[[dict[str, Any]], None], fragment: str
) -> None:
    tree, layout = _layout(tmp_path)
    mapping = batch_mapping(tree)
    change(mapping)
    path = write_batch_config(tree.inputs, mapping)
    error = _refusal(lambda: prepare_batch(str(path), layout))
    assert error.code == "plugins_disabled"
    assert fragment in error.message


def test_a_configuration_without_plugins_uses_none(tmp_path: Path) -> None:
    tree, _ = _layout(tmp_path)
    path = write_batch_config(tree.inputs, batch_mapping(tree))
    assert plugin_uses(load_batch_config(path)) == ()


def test_an_invalid_configuration_is_reported_without_the_server_path(tmp_path: Path) -> None:
    tree, layout = _layout(tmp_path)
    path = tree.inputs / "broken.yaml"
    path.write_text("dataset: [unclosed\n", encoding="utf-8")
    error = _refusal(lambda: prepare_batch(str(path), layout))
    assert error.code == "invalid_config"
    assert str(tree.root) not in error.message
    assert "broken.yaml" in error.message


def test_an_unknown_configuration_field_is_named_in_the_details(tmp_path: Path) -> None:
    tree, layout = _layout(tmp_path)
    path = write_batch_config(tree.inputs, batch_mapping(tree, bogus_setting=1))
    error = _refusal(lambda: prepare_batch(str(path), layout))
    assert error.code == "invalid_config"
    assert any("bogus_setting" in detail for detail in error.details)


def test_relative_paths_are_refused_with_a_clear_message(tmp_path: Path) -> None:
    _, layout = _layout(tmp_path)
    error = _refusal(lambda: prepare_batch("batch.yaml", layout))
    assert error.code == "bad_request"
    assert "absolute" in error.message


def test_a_statistics_design_is_validated_and_summarised(tmp_path: Path) -> None:
    tree, layout = _layout(tmp_path)
    write_measurements(tree.inputs)
    write_dictionary(tree.inputs)
    mapping = design_mapping(tree.inputs, output_directory=str(tree.outputs / "stats"))
    summary = prepare_stats(str(write_design(tree.inputs, mapping)), layout)
    assert summary["groups"] == {"control": 2, "treated": 2}
    assert summary["object_rows"] == 12
    assert summary["output"] == "output/stats"


def test_a_statistics_design_whose_output_is_outside_the_roots_is_refused(tmp_path: Path) -> None:
    tree, layout = _layout(tmp_path)
    write_measurements(tree.inputs)
    write_dictionary(tree.inputs)
    mapping = design_mapping(tree.inputs, output_directory=str(tree.root / "x"))
    path = write_design(tree.inputs, mapping)
    assert _refusal(lambda: prepare_stats(str(path), layout)).code == "path_not_allowed"


def test_a_statistics_design_with_a_bad_setting_is_invalid(tmp_path: Path) -> None:
    tree, layout = _layout(tmp_path)
    write_measurements(tree.inputs)
    mapping = design_mapping(
        tree.inputs, output_directory=str(tree.outputs / "s"), correction="bonferroni"
    )
    error = _refusal(lambda: prepare_stats(str(write_design(tree.inputs, mapping)), layout))
    assert error.code == "invalid_config"
    assert "correction" in " ".join([error.message, *error.details])

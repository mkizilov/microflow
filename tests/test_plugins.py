import textwrap
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from microflow.plugins.api import (
    ENTRY_POINT_GROUPS,
    KINDS,
    FeatureColumn,
    PluginError,
    PluginSpec,
)
from microflow.plugins.contracts import (
    call,
    check_feature_declaration,
    check_features,
    check_findings,
    check_image,
    check_labels,
)
from microflow.plugins.registry import configure, discover, load

SEGMENTER_GROUP = ENTRY_POINT_GROUPS["segmenter"]


def segmenter_module(
    *,
    name: str = "good-seg",
    version: str = "2.0",
    api: str = "PLUGIN_API_VERSION",
    kind: str = "segmenter",
    extra: str = "forbid",
    prelude: str = "",
    body: str = "return (plane > options.level).astype(np.int32)",
) -> str:
    return textwrap.dedent(
        f"""
        import numpy as np
        from pydantic import BaseModel, ConfigDict
        from microflow.plugins.api import PLUGIN_API_VERSION
        {prelude}

        class Options(BaseModel):
            model_config = ConfigDict(extra="{extra}")
            level: float = 0.5

        class Segmenter:
            name = "{name}"
            version = "{version}"
            api_version = {api}
            kind = "{kind}"
            options_model = Options

            def segment(self, plane, calibration, options):
                {body}
        """
    )


# --- declaration and discovery ----------------------------------------------------------------


def test_plugin_names_must_be_lower_case_identifiers() -> None:
    assert PluginSpec(name="edge-detect_2").name == "edge-detect_2"
    with pytest.raises(ValidationError, match="lower-case"):
        PluginSpec(name="Edge Detect")
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        PluginSpec.model_validate({"name": "ok", "option": {}})


def test_every_kind_has_one_entry_point_group() -> None:
    assert set(ENTRY_POINT_GROUPS) == set(KINDS)
    assert len(set(ENTRY_POINT_GROUPS.values())) == len(KINDS)


def test_discovery_reads_metadata_and_runs_no_plugin_code(
    plugin_site: Callable[..., None], tmp_path: Path
) -> None:
    marker = tmp_path / "imported.marker"
    source = segmenter_module(name="marker-seg", prelude=f'open({str(marker)!r}, "w").close()')
    plugin_site(
        "marker-dist",
        {SEGMENTER_GROUP: {"marker-seg": "mf_marker_seg:Segmenter"}},
        {"mf_marker_seg.py": source},
    )

    found = [item for item in discover("segmenter") if item.name == "marker-seg"]
    assert [item.distribution for item in found] == ["marker-dist"]
    assert found[0].target == "mf_marker_seg:Segmenter"
    assert not marker.exists()

    assert load("segmenter", "marker-seg").name == "marker-seg"
    assert marker.exists()


def test_unknown_kind_is_refused() -> None:
    with pytest.raises(PluginError, match="unknown plugin kind 'nonsense'"):
        discover("nonsense")


def test_missing_plugin_error_lists_what_is_installed(plugin_site: Callable[..., None]) -> None:
    plugin_site(
        "present-dist",
        {SEGMENTER_GROUP: {"present-seg": "mf_present:Segmenter"}},
        {"mf_present.py": segmenter_module(name="present-seg")},
    )
    with pytest.raises(PluginError) as caught:
        load("segmenter", "absent-seg")
    message = str(caught.value)
    assert "no segmenter plugin named 'absent-seg' is installed" in message
    assert "present-seg" in message


def test_two_distributions_providing_one_name_are_refused(
    plugin_site: Callable[..., None],
) -> None:
    plugin_site(
        "first-dist",
        {SEGMENTER_GROUP: {"dup-seg": "mf_first:Segmenter"}},
        {"mf_first.py": segmenter_module(name="dup-seg")},
    )
    plugin_site(
        "second-dist",
        {SEGMENTER_GROUP: {"dup-seg": "mf_second:Segmenter"}},
        {"mf_second.py": segmenter_module(name="dup-seg")},
    )
    with pytest.raises(PluginError, match="more than one distribution") as caught:
        load("segmenter", "dup-seg")
    assert "first-dist" in str(caught.value) and "second-dist" in str(caught.value)


# --- load-time validation of the declaration ---------------------------------------------------


@pytest.mark.parametrize(
    ("module_kwargs", "expected"),
    [
        ({"api": "PLUGIN_API_VERSION + 1"}, "was written for plugin API"),
        ({"kind": "feature"}, "declares kind 'feature'; it is registered as 'segmenter'"),
        ({"name": "other-seg"}, "the name must equal its entry-point name"),
        ({"version": ""}, "must declare a non-empty text version"),
        ({"extra": "ignore"}, "must set extra='forbid'"),
        ({"extra": "allow"}, "must set extra='forbid'"),
    ],
)
def test_declarations_that_break_the_interface_are_refused(
    plugin_site: Callable[..., None], module_kwargs: dict[str, str], expected: str
) -> None:
    plugin_site(
        "bad-dist",
        {SEGMENTER_GROUP: {"good-seg": "mf_bad:Segmenter"}},
        {"mf_bad.py": segmenter_module(**module_kwargs)},
    )
    with pytest.raises(PluginError, match=expected):
        load("segmenter", "good-seg")


def test_a_class_without_the_required_method_is_refused(plugin_site: Callable[..., None]) -> None:
    source = textwrap.dedent(
        """
        from pydantic import BaseModel, ConfigDict
        from microflow.plugins.api import PLUGIN_API_VERSION

        class Options(BaseModel):
            model_config = ConfigDict(extra="forbid")

        class Segmenter:
            name = "half-seg"
            version = "1.0"
            api_version = PLUGIN_API_VERSION
            kind = "segmenter"
            options_model = Options
        """
    )
    plugin_site(
        "half-dist",
        {SEGMENTER_GROUP: {"half-seg": "mf_half:Segmenter"}},
        {"mf_half.py": source},
    )
    with pytest.raises(PluginError, match=r"does not implement the segmenter interface.*segment"):
        load("segmenter", "half-seg")


def test_options_model_must_be_a_pydantic_model(plugin_site: Callable[..., None]) -> None:
    source = textwrap.dedent(
        """
        from microflow.plugins.api import PLUGIN_API_VERSION

        class Segmenter:
            name = "plain-seg"
            version = "1.0"
            api_version = PLUGIN_API_VERSION
            kind = "segmenter"
            options_model = dict

            def segment(self, plane, calibration, options):
                return plane
        """
    )
    plugin_site(
        "plain-dist",
        {SEGMENTER_GROUP: {"plain-seg": "mf_plain:Segmenter"}},
        {"mf_plain.py": source},
    )
    with pytest.raises(PluginError, match="must be a pydantic BaseModel subclass"):
        load("segmenter", "plain-seg")


def test_an_import_failure_names_the_plugin(plugin_site: Callable[..., None]) -> None:
    source = segmenter_module(name="broken-seg", prelude='raise RuntimeError("missing data file")')
    plugin_site(
        "broken-dist",
        {SEGMENTER_GROUP: {"broken-seg": "mf_broken:Segmenter"}},
        {"mf_broken.py": source},
    )
    with pytest.raises(PluginError) as caught:
        load("segmenter", "broken-seg")
    message = str(caught.value)
    assert "could not be imported" in message
    assert "RuntimeError: missing data file" in message
    assert "broken-dist" in message


def test_a_constructor_failure_names_the_plugin(plugin_site: Callable[..., None]) -> None:
    source = textwrap.dedent(
        """
        from microflow.plugins.api import PLUGIN_API_VERSION

        class Segmenter:
            name = "ctor-seg"
            version = "1.0"
            api_version = PLUGIN_API_VERSION
            kind = "segmenter"

            def __init__(self):
                raise RuntimeError("no model weights")
        """
    )
    plugin_site(
        "ctor-dist",
        {SEGMENTER_GROUP: {"ctor-seg": "mf_ctor:Segmenter"}},
        {"mf_ctor.py": source},
    )
    with pytest.raises(PluginError, match="could not be created: RuntimeError: no model weights"):
        load("segmenter", "ctor-seg")


def test_feature_declarations_are_checked_at_load(plugin_site: Callable[..., None]) -> None:
    source = textwrap.dedent(
        """
        from pydantic import BaseModel, ConfigDict
        from microflow.plugins.api import PLUGIN_API_VERSION, FeatureColumn

        class Options(BaseModel):
            model_config = ConfigDict(extra="forbid")

        class Feature:
            name = "bad-feat"
            version = "1.0"
            api_version = PLUGIN_API_VERSION
            kind = "feature"
            options_model = Options

            def features(self):
                return (FeatureColumn(name="Bad Name", unit="um", description="x"),)

            def measure(self, labels, plane, calibration, options):
                return {}
        """
    )
    plugin_site(
        "feat-dist",
        {ENTRY_POINT_GROUPS["feature"]: {"bad-feat": "mf_feat:Feature"}},
        {"mf_feat.py": source},
    )
    with pytest.raises(PluginError, match="feature names are lower-case"):
        load("feature", "bad-feat")


def test_configure_reports_invalid_options_with_their_field(
    plugin_site: Callable[..., None],
) -> None:
    plugin_site(
        "opt-dist",
        {SEGMENTER_GROUP: {"opt-seg": "mf_opt:Segmenter"}},
        {"mf_opt.py": segmenter_module(name="opt-seg")},
    )
    with pytest.raises(PluginError, match="invalid options: level:"):
        configure("segmenter", PluginSpec(name="opt-seg", options={"level": "high"}))
    with pytest.raises(PluginError, match="invalid options: bogus:"):
        configure("segmenter", PluginSpec(name="opt-seg", options={"bogus": 1}))
    configured = configure("segmenter", PluginSpec(name="opt-seg", options={"level": 0.25}))
    assert configured.options.model_dump() == {"level": 0.25}
    assert configured.version == "2.0"


# --- output contracts --------------------------------------------------------------------------

COLUMNS = (FeatureColumn("area_like", "um2", "a test value"),)


def test_image_contract_checks_shape_dtype_and_infinity() -> None:
    expected = (2, 2)
    with pytest.raises(PluginError, match=r"returned shape \(3, 2\)"):
        check_image(np.zeros((3, 2)), expected, "p")
    with pytest.raises(PluginError, match="returned dtype bool"):
        check_image(np.zeros(expected, dtype=bool), expected, "p")
    with pytest.raises(PluginError, match="infinite values"):
        check_image(np.array([[1.0, np.inf], [0.0, 0.0]]), expected, "p")
    kept = check_image(np.array([[1, np.nan], [0, 2]], dtype=np.float32), expected, "p")
    assert kept.dtype == np.float64
    assert np.isnan(kept[0, 1])


def test_label_contract_requires_background_zero_and_bounded_integers() -> None:
    expected = (2, 2)
    with pytest.raises(PluginError, match="returned shape"):
        check_labels(np.zeros((2, 3), dtype=np.int32), expected, "s")
    with pytest.raises(PluginError, match="integer labels are required"):
        check_labels(np.zeros(expected, dtype=float), expected, "s")
    with pytest.raises(PluginError, match="negative labels"):
        check_labels(np.array([[0, -1], [0, 0]]), expected, "s")
    with pytest.raises(PluginError, match="exceeds the number of pixels"):
        check_labels(np.array([[0, 5], [0, 0]]), expected, "s")
    labels = check_labels(np.array([[0, 1], [1, 0]], dtype=np.int64), expected, "s")
    assert labels.dtype == np.int32


def test_feature_declarations_must_be_complete_and_unique() -> None:
    with pytest.raises(PluginError, match="declares no features"):
        check_feature_declaration((), "f")
    with pytest.raises(PluginError, match="more than once"):
        check_feature_declaration(
            (FeatureColumn("a", "um", "x"), FeatureColumn("a", "um", "y")), "f"
        )
    with pytest.raises(PluginError, match="must give a unit and a description"):
        check_feature_declaration((FeatureColumn("a", " ", "x"),), "f")
    with pytest.raises(PluginError, match="not a FeatureColumn"):
        check_feature_declaration(("a",), "f")


def test_feature_names_cannot_overwrite_the_row_identity_columns() -> None:
    for reserved in ("label", "plane"):
        with pytest.raises(PluginError, match="are reserved for object identity"):
            check_feature_declaration((FeatureColumn(reserved, "1", "x"),), "f")


def test_feature_values_must_cover_exactly_the_objects_and_declared_features() -> None:
    objects = {1, 2}
    with pytest.raises(PluginError, match=r"no values for object\(s\) \[2\]"):
        check_features({1: {"area_like": 1.0}}, objects, COLUMNS, "f")
    with pytest.raises(PluginError, match=r"not objects"):
        check_features(
            {1: {"area_like": 1.0}, 2: {"area_like": 1.0}, 3: {"area_like": 1.0}},
            objects,
            COLUMNS,
            "f",
        )
    with pytest.raises(PluginError, match="declared features"):
        check_features({1: {"other": 1.0}, 2: {"area_like": 1.0}}, objects, COLUMNS, "f")
    with pytest.raises(PluginError, match="object labels must be integers"):
        check_features({"1": {"area_like": 1.0}, 2: {"area_like": 1.0}}, objects, COLUMNS, "f")
    with pytest.raises(PluginError, match="a number is required"):
        check_features({1: {"area_like": "x"}, 2: {"area_like": 1.0}}, objects, COLUMNS, "f")
    with pytest.raises(PluginError, match="a number is required"):
        check_features({1: {"area_like": True}, 2: {"area_like": 1.0}}, objects, COLUMNS, "f")
    with pytest.raises(PluginError, match="infinite value"):
        check_features({1: {"area_like": np.inf}, 2: {"area_like": 1.0}}, objects, COLUMNS, "f")
    with pytest.raises(PluginError, match="must return a mapping"):
        check_features([1, 2], objects, COLUMNS, "f")

    table = check_features(
        {1: {"area_like": np.float32(2.5)}, 2: {"area_like": float("nan")}},
        objects,
        COLUMNS,
        "f",
    )
    assert table[1] == {"area_like": 2.5}
    assert np.isnan(table[2]["area_like"])


def test_quality_findings_must_be_non_empty_text() -> None:
    with pytest.raises(PluginError, match="sequence of text messages"):
        check_findings("a single string", "q")
    with pytest.raises(PluginError, match="empty or non-text"):
        check_findings(["   "], "q")
    with pytest.raises(PluginError, match="empty or non-text"):
        check_findings([3], "q")
    assert check_findings([], "q") == ()
    assert check_findings(["  coverage high  "], "q") == ("coverage high",)


def test_plugin_exceptions_are_named_and_interrupts_are_not_swallowed() -> None:
    def raises_value_error(value: int) -> int:
        raise ValueError(f"bad value {value}")

    with pytest.raises(PluginError) as caught:
        call("segmenter", "x", "segment", raises_value_error, 7)
    expected = "segmenter plugin 'x' failed during segment: ValueError: bad value 7"
    assert str(caught.value) == expected
    assert isinstance(caught.value.__cause__, ValueError)

    def exits() -> None:
        raise SystemExit(3)

    with pytest.raises(PluginError, match="SystemExit"):
        call("preprocessor", "y", "process", exits)

    def interrupted() -> None:
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        call("feature", "z", "measure", interrupted)

    def inner() -> None:
        raise PluginError("already explained")

    with pytest.raises(PluginError, match=r"^already explained$"):
        call("qc", "w", "check", inner)

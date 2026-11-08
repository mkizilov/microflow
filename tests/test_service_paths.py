import os
from collections.abc import Callable
from pathlib import Path

import pytest

from microflow.service.errors import ServiceError
from microflow.service.paths import (
    list_result_files,
    redact,
    resolve_input,
    resolve_output,
    resolve_result_file,
)

Roots = tuple[Path, Path, Path]


@pytest.fixture
def roots(tmp_path: Path) -> Roots:
    """Input root, output root and a directory outside both of them."""
    inputs = (tmp_path / "in").resolve()
    outputs = (tmp_path / "out").resolve()
    outside = (tmp_path / "outside").resolve()
    for folder in (inputs, outputs, outside):
        folder.mkdir()
    return inputs, outputs, outside


def _code(call: Callable[[], object]) -> str:
    with pytest.raises(ServiceError) as caught:
        call()
    return caught.value.code


def _run_directory(root: Path) -> Path:
    run = root / "run"
    (run / "items").mkdir(parents=True)
    (run / "summary.csv").write_text("a\n", encoding="utf-8")
    (run / "items" / "labels.tif").write_bytes(b"x")
    (run / "nested").mkdir()
    (run / "nested" / "notes.json").write_text("{}", encoding="utf-8")
    return run


def test_an_input_inside_a_root_resolves_to_its_real_path(roots: Roots) -> None:
    inputs, _, _ = roots
    (inputs / "image.tif").write_bytes(b"x")
    assert resolve_input(str(inputs / "image.tif"), (inputs,)) == inputs / "image.tif"


def test_inputs_outside_the_roots_are_refused_whatever_their_spelling(roots: Roots) -> None:
    inputs, _, outside = roots
    (outside / "secret.tif").write_bytes(b"x")
    direct = str(outside / "secret.tif")
    traversal = f"{inputs}/../outside/secret.tif"
    assert _code(lambda: resolve_input(direct, (inputs,))) == "path_not_allowed"
    assert _code(lambda: resolve_input(traversal, (inputs,))) == "path_not_allowed"


def test_a_symbolic_link_that_leads_outside_is_refused(roots: Roots) -> None:
    inputs, _, outside = roots
    (outside / "secret.tif").write_bytes(b"x")
    os.symlink(outside / "secret.tif", inputs / "link.tif")
    link = str(inputs / "link.tif")
    assert _code(lambda: resolve_input(link, (inputs,))) == "path_not_allowed"


def test_a_symbolic_link_that_stays_inside_is_allowed(roots: Roots) -> None:
    inputs, _, _ = roots
    (inputs / "real.tif").write_bytes(b"x")
    os.symlink(inputs / "real.tif", inputs / "alias.tif")
    assert resolve_input(str(inputs / "alias.tif"), (inputs,)) == inputs / "real.tif"


def test_a_missing_input_inside_a_root_is_not_found(roots: Roots) -> None:
    inputs, _, _ = roots
    absent = str(inputs / "absent.tif")
    assert _code(lambda: resolve_input(absent, (inputs,))) == "not_found"


@pytest.mark.parametrize(
    "value",
    ["relative/path.tif", "", None, 42, "bad\x00name.tif", "/" + "a" * 5000],
)
def test_malformed_paths_are_bad_requests(roots: Roots, value: object) -> None:
    inputs, _, _ = roots
    assert _code(lambda: resolve_input(value, (inputs,))) == "bad_request"


def test_an_output_inside_an_output_root_may_not_exist_yet(roots: Roots) -> None:
    _, outputs, _ = roots
    target = outputs / "new-run"
    assert resolve_output(str(target), (outputs,)) == target


def test_an_output_may_not_be_the_root_itself_or_outside_it(roots: Roots) -> None:
    _, outputs, outside = roots
    root_text = str(outputs)
    outside_text = str(outside / "run")
    escape = str(outputs / ".." / "outside" / "run")
    assert _code(lambda: resolve_output(root_text, (outputs,))) == "path_not_allowed"
    assert _code(lambda: resolve_output(outside_text, (outputs,))) == "path_not_allowed"
    assert _code(lambda: resolve_output(escape, (outputs,))) == "path_not_allowed"


def test_an_output_that_is_an_existing_file_is_refused(roots: Roots) -> None:
    _, outputs, _ = roots
    (outputs / "file.csv").write_text("x", encoding="utf-8")
    existing = str(outputs / "file.csv")
    assert _code(lambda: resolve_output(existing, (outputs,))) == "bad_request"


def test_a_result_file_is_served_by_its_relative_name(roots: Roots) -> None:
    _, outputs, _ = roots
    run = _run_directory(outputs)
    assert resolve_result_file(run, "summary.csv") == run / "summary.csv"
    assert resolve_result_file(run, "items/labels.tif") == run / "items" / "labels.tif"


@pytest.mark.parametrize(
    ("name", "code"),
    [
        ("../outside.csv", "path_not_allowed"),
        ("nested/../../outside.csv", "path_not_allowed"),
        ("./summary.csv", "path_not_allowed"),
        (".hidden.csv", "path_not_allowed"),
        ("items/.partial.tif", "path_not_allowed"),
        ("/summary.csv", "bad_request"),
        ("nested\\notes.json", "bad_request"),
        ("nested//notes.json", "bad_request"),
        ("", "bad_request"),
        ("secret.txt", "path_not_allowed"),
        ("nested", "path_not_allowed"),
        ("missing.csv", "not_found"),
        (None, "bad_request"),
    ],
)
def test_names_that_escape_hide_or_are_not_results_are_refused(
    roots: Roots, name: object, code: str
) -> None:
    _, outputs, _ = roots
    run = _run_directory(outputs)
    (run / "secret.txt").write_text("nope", encoding="utf-8")
    (run / ".hidden.csv").write_text("nope", encoding="utf-8")
    (outputs / "outside.csv").write_text("nope", encoding="utf-8")
    assert _code(lambda: resolve_result_file(run, name)) == code


def test_a_symbolic_link_inside_a_run_is_not_served(roots: Roots) -> None:
    _, outputs, outside = roots
    run = _run_directory(outputs)
    (outside / "secret.csv").write_text("nope", encoding="utf-8")
    os.symlink(outside / "secret.csv", run / "link.csv")
    os.symlink(outside, run / "linked-dir")
    assert _code(lambda: resolve_result_file(run, "link.csv")) == "path_not_allowed"
    assert _code(lambda: resolve_result_file(run, "linked-dir/secret.csv")) == "path_not_allowed"


def test_listing_shows_only_result_files_and_never_links_or_hidden_names(roots: Roots) -> None:
    _, outputs, outside = roots
    run = _run_directory(outputs)
    (run / "secret.txt").write_text("nope", encoding="utf-8")
    (run / ".hidden.csv").write_text("nope", encoding="utf-8")
    (run / "labels.tif.partial").write_bytes(b"half")
    (outside / "secret.csv").write_text("nope", encoding="utf-8")
    os.symlink(outside / "secret.csv", run / "link.csv")
    os.symlink(outside, run / "linked-dir")
    files, truncated = list_result_files(run)
    assert [item.path for item in files] == ["items/labels.tif", "nested/notes.json", "summary.csv"]
    assert files[2].size_bytes == 2
    assert not truncated


def test_listing_is_bounded_and_says_when_it_was_cut_short(roots: Roots) -> None:
    _, outputs, _ = roots
    run = outputs / "many"
    run.mkdir()
    for index in range(5):
        (run / f"file{index}.csv").write_text("x", encoding="utf-8")
    files, truncated = list_result_files(run, limit=3)
    assert [item.path for item in files] == ["file0.csv", "file1.csv", "file2.csv"]
    assert truncated


def test_redaction_replaces_the_longest_path_first(tmp_path: Path) -> None:
    text = f"{tmp_path}/output/run-1/x and {tmp_path}/output"
    replacements = {
        str(tmp_path / "output"): "out",
        str(tmp_path / "output" / "run-1"): "run",
    }
    assert redact(text, replacements) == "run/x and out"

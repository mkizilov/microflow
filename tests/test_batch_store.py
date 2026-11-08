import os
import subprocess
import sys
from pathlib import Path

import pytest

from microflow.batch import store
from microflow.batch.records import ItemRecord, OutputFile
from microflow.batch.store import BatchError


def _valid_item(item_dir: Path, *, fingerprint: str = "f", input_sha: str = "in") -> ItemRecord:
    item_dir.mkdir(parents=True)
    (item_dir / "labels.tif").write_bytes(b"labels")
    output = OutputFile(
        path="labels.tif",
        sha256=store.sha256_file(item_dir / "labels.tif"),
        size_bytes=6,
    )
    record = ItemRecord(
        name="a",
        status="ok",
        input_sha256=input_sha,
        config_fingerprint=fingerprint,
        outputs={"labels": output},
    )
    store.write_result(item_dir, record)
    return record


def test_valid_item_is_reused_and_marked_resumed(tmp_path: Path) -> None:
    item_dir = tmp_path / "item"
    _valid_item(item_dir)
    cached, reason = store.check_cache(item_dir, "in", "f")
    assert cached is not None
    assert cached.resumed is True
    assert reason == store.CACHED


def test_missing_result_is_not_reused(tmp_path: Path) -> None:
    assert store.check_cache(tmp_path / "item", "in", "f") == (None, store.NO_PREVIOUS)


def test_failed_result_is_not_reused(tmp_path: Path) -> None:
    item_dir = tmp_path / "item"
    item_dir.mkdir()
    store.write_result(item_dir, ItemRecord(name="a", status="failed", input_sha256="in"))
    assert store.check_cache(item_dir, "in", "f") == (None, "previous run failed")


def test_changed_input_is_not_reused(tmp_path: Path) -> None:
    item_dir = tmp_path / "item"
    _valid_item(item_dir)
    assert store.check_cache(item_dir, "other", "f") == (None, "input changed")


def test_changed_configuration_is_not_reused(tmp_path: Path) -> None:
    item_dir = tmp_path / "item"
    _valid_item(item_dir)
    assert store.check_cache(item_dir, "in", "other") == (None, "configuration changed")


def test_missing_output_is_not_reused(tmp_path: Path) -> None:
    item_dir = tmp_path / "item"
    _valid_item(item_dir)
    (item_dir / "labels.tif").unlink()
    assert store.check_cache(item_dir, "in", "f") == (None, "output labels.tif is missing")


def test_modified_output_is_not_reused(tmp_path: Path) -> None:
    item_dir = tmp_path / "item"
    _valid_item(item_dir)
    (item_dir / "labels.tif").write_bytes(b"LABELS")
    assert store.check_cache(item_dir, "in", "f") == (None, "output labels.tif was modified")


def test_truncated_output_is_not_reused(tmp_path: Path) -> None:
    item_dir = tmp_path / "item"
    _valid_item(item_dir)
    (item_dir / "labels.tif").write_bytes(b"lab")
    assert store.check_cache(item_dir, "in", "f") == (None, "output labels.tif was modified")


def test_unreadable_result_is_not_reused(tmp_path: Path) -> None:
    item_dir = tmp_path / "item"
    item_dir.mkdir()
    (item_dir / store.RESULT_FILE).write_text("{not json", encoding="utf-8")
    assert store.check_cache(item_dir, "in", "f") == (None, "previous result unreadable")


def test_write_output_publishes_only_a_completed_file(tmp_path: Path) -> None:
    def writer(path: Path) -> None:
        path.write_bytes(b"complete")

    output = store.write_output(tmp_path, "data.bin", writer)
    assert (tmp_path / "data.bin").read_bytes() == b"complete"
    assert not (tmp_path / "data.bin.partial").exists()
    assert output.size_bytes == 8
    assert output.sha256 == store.sha256_file(tmp_path / "data.bin")


def test_failed_writer_leaves_no_final_file(tmp_path: Path) -> None:
    def writer(path: Path) -> None:
        path.write_bytes(b"half")
        raise RuntimeError("disk full")

    with pytest.raises(RuntimeError, match="disk full"):
        store.write_output(tmp_path, "data.bin", writer)
    assert not (tmp_path / "data.bin").exists()


def test_item_directory_names_are_safe_and_stay_distinct() -> None:
    assert "/" not in store.item_directory_name("a/b")
    assert store.item_directory_name("a b") != store.item_directory_name("a_b")
    assert store.item_directory_name("img_1") == store.item_directory_name("img_1")


def test_digest_does_not_depend_on_key_order() -> None:
    assert store.digest({"a": 1, "b": [2]}) == store.digest({"b": [2], "a": 1})


def test_reset_removes_final_markers_and_partial_files_but_keeps_results(tmp_path: Path) -> None:
    item_dir = tmp_path / "items" / "x"
    _valid_item(item_dir)
    (item_dir / "labels.tif.partial").write_bytes(b"x")
    (tmp_path / store.MANIFEST_FILE).write_text("{}", encoding="utf-8")
    (tmp_path / store.CANCEL_FILE).write_text("", encoding="utf-8")
    store.reset_run_directory(tmp_path)
    assert not (item_dir / "labels.tif.partial").exists()
    assert not (tmp_path / store.MANIFEST_FILE).exists()
    assert not (tmp_path / store.CANCEL_FILE).exists()
    assert (item_dir / store.RESULT_FILE).exists()


def test_run_lock_is_released_after_use(tmp_path: Path) -> None:
    with store.run_lock(tmp_path):
        assert (tmp_path / store.LOCK_FILE).exists()
    assert not (tmp_path / store.LOCK_FILE).exists()


def test_run_lock_refuses_a_live_holder(tmp_path: Path) -> None:
    lock = tmp_path / store.LOCK_FILE
    lock.write_text(str(os.getpid()), encoding="utf-8")
    with pytest.raises(BatchError, match="another batch run"):
        with store.run_lock(tmp_path):
            pass
    assert lock.read_text(encoding="utf-8") == str(os.getpid())


def test_run_lock_takes_over_a_lock_left_by_a_dead_process(tmp_path: Path) -> None:
    process = subprocess.Popen([sys.executable, "-c", "pass"])
    process.wait()
    (tmp_path / store.LOCK_FILE).write_text(str(process.pid), encoding="utf-8")
    with store.run_lock(tmp_path):
        assert (tmp_path / store.LOCK_FILE).read_text(encoding="utf-8") == str(os.getpid())
    assert not (tmp_path / store.LOCK_FILE).exists()

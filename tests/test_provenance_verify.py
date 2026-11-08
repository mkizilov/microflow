import hashlib
import json
from pathlib import Path

import pytest

from microflow.provenance.verify import VerificationError, sha256_of, verify_run


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _run(root: Path) -> Path:
    """A finished run with one input image and one item output, laid out as batch writes them."""
    raw = root / "raw.tif"
    raw.write_bytes(b"raw image bytes")
    run = root / "run"
    item = run / "items" / "sample-1234abcd"
    item.mkdir(parents=True)
    (item / "labels.tif").write_bytes(b"label bytes")
    result = {
        "name": "sample",
        "status": "ok",
        "outputs": {
            "labels": {
                "path": "labels.tif",
                "sha256": _digest(b"label bytes"),
                "size_bytes": len(b"label bytes"),
            }
        },
    }
    (item / "result.json").write_text(json.dumps(result), encoding="utf-8")
    manifest = {
        "status": "complete",
        "provenance": {
            "inputs": [
                {
                    "name": "sample",
                    "path": str(raw),
                    "sha256": _digest(b"raw image bytes"),
                    "size_bytes": len(b"raw image bytes"),
                }
            ]
        },
    }
    (run / "batch_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return run


def test_an_untouched_run_verifies_every_recorded_file(tmp_path: Path) -> None:
    verification = verify_run(_run(tmp_path))
    assert verification.ok
    assert verification.run_status == "complete"
    assert sorted(check.kind for check in verification.checks) == ["input", "output"]
    assert all(check.status == "ok" for check in verification.checks)


def test_an_edited_input_is_reported_as_modified(tmp_path: Path) -> None:
    run = _run(tmp_path)
    (tmp_path / "raw.tif").write_bytes(b"edited after the run")
    verification = verify_run(run)
    assert not verification.ok
    failed = verification.failed
    assert [(check.kind, check.status) for check in failed] == [("input", "modified")]


def test_a_deleted_output_is_reported_as_missing(tmp_path: Path) -> None:
    run = _run(tmp_path)
    (run / "items" / "sample-1234abcd" / "labels.tif").unlink()
    verification = verify_run(run)
    assert [(check.kind, check.status) for check in verification.failed] == [("output", "missing")]
    assert verification.failed[0].name == "sample/labels"


def test_a_run_without_a_final_manifest_cannot_be_verified(tmp_path: Path) -> None:
    run = _run(tmp_path)
    (run / "batch_manifest.json").unlink()
    with pytest.raises(VerificationError, match="only finished runs"):
        verify_run(run)


def test_a_corrupt_manifest_is_reported_not_crashed_on(tmp_path: Path) -> None:
    run = _run(tmp_path)
    (run / "batch_manifest.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(VerificationError, match="cannot read the run manifest"):
        verify_run(run)


def test_the_hash_matches_the_standard_digest_of_known_bytes(tmp_path: Path) -> None:
    path = tmp_path / "known.bin"
    path.write_bytes(b"abc")
    assert sha256_of(path) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"

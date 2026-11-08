from pathlib import Path

import pytest
from pydantic import ValidationError

from microflow.config import ConfigError
from microflow.service.settings import (
    DATABASE_NAME,
    MIN_TOKEN_LENGTH,
    TOKEN_ENV,
    check_bind,
    is_loopback,
    load_settings,
    prepare_layout,
    token_from_environment,
)
from service_fixtures import make_tree, settings_for


def test_loopback_addresses_are_recognised() -> None:
    assert is_loopback("127.0.0.1")
    assert is_loopback("::1")
    assert is_loopback("localhost")
    assert not is_loopback("0.0.0.0")
    assert not is_loopback("192.168.1.10")
    assert not is_loopback("example.com")


def test_remote_bind_is_refused_unless_explicitly_allowed(tmp_path: Path) -> None:
    tree = make_tree(tmp_path)
    with pytest.raises(ConfigError, match=r"refusing to listen on 0.0.0.0"):
        check_bind("0.0.0.0", allow_remote=False)
    check_bind("0.0.0.0", allow_remote=True)
    with pytest.raises(ValidationError, match="refusing to listen"):
        settings_for(tree, host="0.0.0.0")
    assert settings_for(tree, host="0.0.0.0", allow_remote=True).host == "0.0.0.0"


def test_directories_must_be_absolute_and_listed_once(tmp_path: Path) -> None:
    tree = make_tree(tmp_path)
    with pytest.raises(ValidationError, match="absolute path"):
        settings_for(tree, input_roots=("relative/inputs",))
    with pytest.raises(ValidationError, match="absolute path"):
        settings_for(tree, state_directory="state")
    with pytest.raises(ValidationError, match="listed more than once"):
        settings_for(tree, input_roots=(tree.inputs, tree.inputs))


def test_limits_are_bounded(tmp_path: Path) -> None:
    tree = make_tree(tmp_path)
    with pytest.raises(ValidationError):
        settings_for(tree, max_concurrent_jobs=0)
    with pytest.raises(ValidationError):
        settings_for(tree, port=70000)
    with pytest.raises(ValidationError):
        settings_for(tree, max_queued_jobs=257)


def test_a_settings_file_resolves_relative_paths_against_its_own_directory(tmp_path: Path) -> None:
    config_dir = tmp_path / "service"
    config_dir.mkdir()
    path = config_dir / "service.yaml"
    path.write_text(
        "state_directory: state\n"
        "input_roots: [inputs]\n"
        "output_roots: [outputs]\n"
        "max_concurrent_jobs: 2\n",
        encoding="utf-8",
    )
    settings = load_settings(path)
    assert settings.state_directory == config_dir / "state"
    assert settings.input_roots == (config_dir / "inputs",)
    assert settings.max_concurrent_jobs == 2


def test_an_unknown_setting_is_reported_with_its_name(tmp_path: Path) -> None:
    path = tmp_path / "service.yaml"
    path.write_text(
        "state_directory: s\ninput_roots: [i]\noutput_roots: [o]\nsecret_key: nope\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="secret_key"):
        load_settings(path)


def test_the_token_comes_from_the_environment_and_is_never_echoed() -> None:
    short = "x" * (MIN_TOKEN_LENGTH - 1)
    with pytest.raises(ConfigError, match=TOKEN_ENV) as missing:
        token_from_environment({})
    assert short not in str(missing.value)
    with pytest.raises(ConfigError) as too_short:
        token_from_environment({TOKEN_ENV: short})
    assert short not in str(too_short.value)
    long_enough = "y" * MIN_TOKEN_LENGTH
    assert token_from_environment({TOKEN_ENV: long_enough}) == long_enough


def test_start_up_refuses_missing_directories(tmp_path: Path) -> None:
    tree = make_tree(tmp_path)
    missing = settings_for(tree, input_roots=(tmp_path / "absent",))
    with pytest.raises(ConfigError, match="input directory"):
        prepare_layout(missing)


def test_the_state_directory_may_not_sit_inside_an_output_directory(tmp_path: Path) -> None:
    tree = make_tree(tmp_path)
    inside = settings_for(tree, state_directory=tree.outputs / "state")
    with pytest.raises(ConfigError, match="must not be inside an output directory"):
        prepare_layout(inside)


def test_start_up_creates_the_state_directory_and_names_the_database(tmp_path: Path) -> None:
    tree = make_tree(tmp_path)
    layout = prepare_layout(settings_for(tree))
    assert layout.state_path == tree.state.resolve() / DATABASE_NAME
    assert tree.state.is_dir()
    assert layout.input_roots == (tree.inputs.resolve(),)
    assert layout.output_roots == (tree.outputs.resolve(),)

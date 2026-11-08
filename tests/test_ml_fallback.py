import numpy as np
import pytest

from microflow.ml.fallback import FallbackRunner


def _zeros(window: np.ndarray) -> np.ndarray:
    return np.zeros(window.shape, dtype=np.float32)


def _is_out_of_memory(exc: Exception) -> bool:
    return "out of memory" in str(exc)


def test_a_recoverable_failure_moves_that_window_and_every_later_one_to_the_fallback() -> None:
    calls = {"primary": 0, "fallback": 0}

    def primary(window: np.ndarray) -> np.ndarray:
        calls["primary"] += 1
        raise RuntimeError("CUDA error: out of memory")

    def fallback(window: np.ndarray) -> np.ndarray:
        calls["fallback"] += 1
        return _zeros(window)

    runner = FallbackRunner(primary, fallback, is_recoverable=_is_out_of_memory)
    runner(np.zeros((2, 2)))
    runner(np.zeros((2, 2)))
    assert runner.fell_back
    assert calls == {"primary": 1, "fallback": 2}
    assert runner.reason is not None and runner.reason.startswith("RuntimeError")


def test_errors_that_are_not_recoverable_are_not_masked() -> None:
    def primary(window: np.ndarray) -> np.ndarray:
        raise ValueError("bad shape")

    runner = FallbackRunner(primary, _zeros, is_recoverable=_is_out_of_memory)
    with pytest.raises(ValueError, match="bad shape"):
        runner(np.zeros((2, 2)))
    assert not runner.fell_back


def test_without_a_fallback_even_a_recoverable_error_propagates() -> None:
    def primary(window: np.ndarray) -> np.ndarray:
        raise RuntimeError("CUDA error: out of memory")

    runner = FallbackRunner(primary, None, is_recoverable=_is_out_of_memory)
    with pytest.raises(RuntimeError, match="out of memory"):
        runner(np.zeros((2, 2)))
    assert not runner.fell_back


def test_a_run_without_failures_never_uses_the_fallback() -> None:
    used: list[str] = []

    def fallback(window: np.ndarray) -> np.ndarray:
        used.append("fallback")
        return _zeros(window)

    runner = FallbackRunner(_zeros, fallback, is_recoverable=_is_out_of_memory)
    for _ in range(3):
        runner(np.zeros((2, 2)))
    assert used == []
    assert not runner.fell_back
    assert runner.reason is None

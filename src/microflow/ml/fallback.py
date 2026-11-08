"""Run a window predictor on its primary device, falling back to a second one. Torch-free.

Fallback is decided per run, not per window. Once the primary device fails with a recoverable error,
every later window goes to the fallback too, so the run does not retry a device that has just
failed.
The reason is kept for the report.
"""

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

Predict = Callable[[NDArray[np.float64]], NDArray[np.float32]]


def _never(exc: Exception) -> bool:
    return False


class FallbackRunner:
    def __init__(
        self,
        primary: Predict,
        fallback: Predict | None = None,
        *,
        is_recoverable: Callable[[Exception], bool] = _never,
    ) -> None:
        self._primary = primary
        self._fallback = fallback
        self._is_recoverable = is_recoverable
        self._fell_back = False
        self.reason: str | None = None

    @property
    def fell_back(self) -> bool:
        return self._fell_back

    def __call__(self, window: NDArray[np.float64]) -> NDArray[np.float32]:
        if self._fell_back and self._fallback is not None:
            return self._fallback(window)
        try:
            return self._primary(window)
        except Exception as exc:
            if self._fallback is None or not self._is_recoverable(exc):
                raise
            self._fell_back = True
            self.reason = f"{type(exc).__name__}: {exc}"
            return self._fallback(window)

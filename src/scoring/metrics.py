from __future__ import annotations

from typing import Sequence, Union

import numpy as np
import pandas as pd


def rmse(x: Union[pd.DataFrame, Sequence[np.ndarray]], na_rm: bool = True) -> float:
    """
    Igual ao R: retorna NEGATIVO (para maximizar no BO).
    """
    if isinstance(x, pd.DataFrame):
        y = pd.to_numeric(x.iloc[:, 0], errors="coerce").to_numpy(dtype=float)
        yhat = pd.to_numeric(x.iloc[:, 1], errors="coerce").to_numpy(dtype=float)
    else:
        y = np.asarray(x[0], dtype=float)
        yhat = np.asarray(x[1], dtype=float)

    if na_rm:
        ok = ~(np.isnan(y) | np.isnan(yhat))
        y, yhat = y[ok], yhat[ok]

    if y.size == 0:
        return float("nan")

    return float(-np.sqrt(np.mean((y - yhat) ** 2)))


def mape(x: Union[pd.DataFrame, Sequence[np.ndarray]], na_rm: bool = True) -> float:
    """
    Igual ao R: retorna NEGATIVO (para maximizar no BO).
    """
    if isinstance(x, pd.DataFrame):
        y = pd.to_numeric(x.iloc[:, 0], errors="coerce").to_numpy(dtype=float)
        yhat = pd.to_numeric(x.iloc[:, 1], errors="coerce").to_numpy(dtype=float)
    else:
        y = np.asarray(x[0], dtype=float)
        yhat = np.asarray(x[1], dtype=float)

    if na_rm:
        ok = ~(np.isnan(y) | np.isnan(yhat))
        y, yhat = y[ok], yhat[ok]

    if y.size == 0:
        return float("nan")

    eps = np.finfo(float).eps
    denom = np.maximum(np.abs(y), eps)
    return float(-np.mean(np.abs(y - yhat) / denom))


def rpe(measured: Union[pd.DataFrame, pd.Series, np.ndarray],
        simulated: Union[pd.DataFrame, pd.Series, np.ndarray]) -> Union[pd.DataFrame, np.ndarray]:
    if isinstance(measured, (pd.DataFrame, pd.Series)) or isinstance(simulated, (pd.DataFrame, pd.Series)):
        measured_df = pd.DataFrame(measured)
        simulated_df = pd.DataFrame(simulated)
        return (simulated_df - measured_df) / measured_df.abs()
    measured_arr = np.asarray(measured, dtype=float)
    simulated_arr = np.asarray(simulated, dtype=float)
    return (simulated_arr - measured_arr) / np.abs(measured_arr)

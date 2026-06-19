from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from .metrics import rmse, mape, rpe


def evaluate_difference(
    evaluate_data: pd.DataFrame,
    calibration: Iterable[str],
    metodo_score: str,
    mape_iqr_lambda: float = 0.5,
) -> float:
    """
    Para cada variável em calibration:
      - tenta usar pares <VAR>M e <VAR>S
      - substitui -99 por NaN
      - aplica rmse/mape (NEGATIVO)
    Retorna a média.
    """
    if not isinstance(evaluate_data, pd.DataFrame):
        evaluate_data = pd.DataFrame(evaluate_data)

    metodo_score = str(metodo_score).lower().strip()
    metodos = {"rmse": rmse, "mape": mape}
    if metodo_score not in metodos:
        raise ValueError("metodo_score inválido. Use 'rmse' ou 'mape'.")

    cols = list(evaluate_data.columns)
    scores = []

    for cal in calibration:
        calu = str(cal).upper()

        obs_candidates = [c for c in cols if str(c).upper() == f"{calu}M"]
        sim_candidates = [c for c in cols if str(c).upper() == f"{calu}S"]
        if not obs_candidates or not sim_candidates:
            # fallback: busca qualquer col que contenha 'cal'
            matched = [c for c in cols if cal in str(c) and str(c) not in ("Origem",)]
            if len(matched) < 2:
                raise ValueError(f"Não encontrei 2 colunas para '{cal}'. Encontradas: {matched}")
            matched = sorted(matched, key=lambda x: str(x))
            obs_col, sim_col = matched[0], matched[1]
        else:
            obs_col, sim_col = obs_candidates[0], sim_candidates[0]
        df2 = evaluate_data[[obs_col, sim_col]].copy().replace(-99, np.nan)

        scores.append(metodos[metodo_score](df2, na_rm=True))

    if metodo_score == "mape":
        mape_values = -np.array(scores, dtype=float)
        median_mape = np.nanmedian(mape_values)
        q75, q25 = np.nanpercentile(mape_values, [75, 25])
        iqr_mape = q75 - q25
        return float(-(median_mape + float(mape_iqr_lambda) * iqr_mape))

    return float(np.nanmean(np.array(scores, dtype=float)))


def plantgro_difference(plantgro_data: pd.DataFrame, t_data: pd.DataFrame, calibration: Sequence[str]) -> pd.DataFrame:
    """
    Porta do seu R plantgroDifference():
      - agrupa por TRNO média
      - merge obs e sim
      - calcula RPE e devolve DataFrame com TN
    """
    calibration = list(calibration)
    needed = ["TRNO"] + calibration

    sim = plantgro_data.loc[:, [c for c in needed if c in plantgro_data.columns]].copy().replace(-99, np.nan)
    sim = sim.groupby("TRNO", as_index=False).mean(numeric_only=True)

    obs = t_data.loc[:, [c for c in needed if c in t_data.columns]].copy().replace(-99, np.nan)
    obs = obs.groupby("TRNO", as_index=False).mean(numeric_only=True)

    merged = obs.merge(sim, on="TRNO", how="inner", suffixes=(".x", ".y")).sort_values("TRNO")

    obs_cols = [f"{c}.x" for c in calibration if f"{c}.x" in merged.columns]
    sim_cols = [f"{c}.y" for c in calibration if f"{c}.y" in merged.columns]

    response = rpe(merged[obs_cols], merged[sim_cols])
    response = pd.DataFrame(response, columns=[c.replace(".x", "").replace(".y", "") for c in obs_cols])
    response["TN"] = merged["TRNO"].to_numpy()
    return response

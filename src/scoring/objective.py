from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence, Optional
from datetime import datetime
import numpy as np
import pandas as pd

from src.utils.logging import append_log
from src.utils.ids import iteration_id_random
from src.dssat.files import create_simulation_directories
from src.dssat.execute import run_dssat


def simulation_function(param_sim: Mapping[str, Any], template_id: str, input_list: Mapping[str, Any]) -> pd.DataFrame:
    """
    Porta do R simulationFunction():
      - cria diretórios e arquivos para a rodada
      - executa DSSAT
      - retorna Evaluate (DataFrame)
    """
    simulation_files = create_simulation_directories(param_sim, template_id, input_list)

    model = str(input_list["model"])
    calibration = list(input_list.get("calibration", []))
    dssat_file = str(input_list["dssatFile"])

    df = run_dssat(
        simulation_files=simulation_files,
        model=model,
        dssat_file=dssat_file,
        calibration=calibration,
        read_treatments_id=input_list["read_treatments_id"],
        read_region=input_list["read_region"],
        read_evaluate=input_list["read_evaluate"],
        cleanup=True,
        crop=str(input_list.get("crop", "BEAN")),
    )
    return df if isinstance(df, pd.DataFrame) else pd.DataFrame()


def scoring_function(
    param_sim: Mapping[str, Any],
    input_list: Mapping[str, Any],
    *,
    simulation_function: Callable[[Mapping[str, Any], str, Mapping[str, Any]], pd.DataFrame],
    evaluate_difference: Callable[[pd.DataFrame, Sequence[str], str], float],
    logfile: str = "output/log_execucao.txt",
    noise_std: float = 1e-4,
) -> float:
    """
    Porta do seu R scoringFunction():
      - loga params
      - roda simulationFunction()
      - calcula Score via evaluateDifference()
      - retorna Score (com ruído muito pequeno)
      - se falhar: retorna -99
    """
    try:
        msg = "Rodando Simulação para os valores:"
        for v in param_sim.values():
            try:
                msg += f" {float(v):.2f}"
            except Exception:
                msg += f" {v}"

        template_id = iteration_id_random("iteration")
        run = simulation_function(param_sim, template_id, input_list)
        metodo_score = str(input_list.get("metodo_score", "rmse")).lower()
        calibration = list(input_list.get("calibration", []))

        if run is None or (isinstance(run, pd.DataFrame) and run.empty):
            score = -99.0
        else:
            score = float(evaluate_difference(run, calibration, metodo_score))

        msg2 = f"{msg} - Valor do Score para rodada: {score:.4f}\n"
        #append_log(logfile, msg2)

        if np.isnan(score):
            return -99.0

        # ruído minúsculo como no R: Score - rnorm(1,0,0.0001)
        if noise_std and noise_std > 0:
            score = score - float(np.random.normal(0.0, noise_std))

        return float(score)

    except Exception:
        return -99.0

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Union

from src.config.parser import config_treatment, load_limites
from src.optimization.bayesopt import run_simulation_baye
from src.optimization.persistence import salvar_resultados_bo
from src.utils.time import calcular_tempo_dec

from src.dssat.cultivar import make_cultivar, write_cultivar
from src.dssat.io import read_treatments_id, read_region, read_evaluate
from src.scoring.difference import evaluate_difference
from src.scoring.objective import simulation_function

from src.analysis.post_training import run_post_training_pipeline_for_bnx

def build_input_list(cfg: Mapping[str, Any]) -> Dict[str, Any]:
    input_list: Dict[str, Any] = dict(cfg)

    for k in ("coefficients", "limites", "calibration"):
        if k in input_list and isinstance(input_list[k], str):
            input_list[k] = [input_list[k]]

    input_list["make_cultivar"] = make_cultivar
    input_list["write_cultivar"] = write_cultivar
    input_list["read_treatments_id"] = read_treatments_id
    input_list["read_region"] = read_region
    input_list["read_evaluate"] = read_evaluate

    input_list.setdefault("crop", "BEAN")
    input_list.setdefault("bnxTargetName", "EMEP1403.BNX")

    if "bnxDir" not in input_list or not input_list["bnxDir"]:
        dir_experiment = Path(str(input_list["dirExperiment"]))
        input_list["bnxDir"] = str(dir_experiment.parent / "Config_BNX")

    return input_list


def _parse_list(value: Any, cast=str) -> List[Any]:
    if isinstance(value, list):
        return [cast(v) for v in value]

    if isinstance(value, tuple):
        return [cast(v) for v in value]

    if isinstance(value, str):
        parts = [p.strip() for p in value.split(",") if p.strip()]
        return [cast(p) for p in parts]

    return [cast(value)]
    
def _get_bnx_pairs(cfg: Mapping[str, Any]) -> List[Dict[str, str]]:
    bnx_files = _parse_list(cfg.get("bnxFiles", []), str)
    bnx_test = _parse_list(cfg.get("bnxTest", []), str)

    if not bnx_files:
        raise ValueError("Nenhum arquivo BNX foi informado em 'bnxFiles'.")
    if not bnx_test:
        raise ValueError("Nenhum arquivo BNX foi informado em 'bnxTest'.")

    if len(bnx_files) != len(bnx_test):
        raise ValueError(
            "As listas 'bnxFiles' e 'bnxTest' devem ter o mesmo tamanho. "
            f"Recebido: {len(bnx_files)} treino vs {len(bnx_test)} teste."
        )

    pairs: List[Dict[str, str]] = []
    for train_bnx, test_bnx in zip(bnx_files, bnx_test, strict=True):
        pairs.append(
            {
                "bnxFile": train_bnx,
                "bnxTest": test_bnx,
                "bnxLabel": Path(train_bnx).stem,
            }
        )
    return pairs


def _build_base_combinacoes(cfg: Mapping[str, Any]) -> List[Dict[str, Any]]:
    pontos_iniciais = _parse_list(cfg.get("initPoints"), int)
    n_iteracoes = _parse_list(cfg.get("iters.n"), int)
    acqs = _parse_list(cfg.get("acq"), str)
    kernels = _parse_list(cfg.get("kernel"), str)

    tamanhos = {
        "initPoints": len(pontos_iniciais),
        "iters.n": len(n_iteracoes),
        "acq": len(acqs),
        "kernel": len(kernels),
    }

    if len(set(tamanhos.values())) != 1:
        raise ValueError(
            "As listas de configuração devem ter o mesmo tamanho. "
            f"Tamanhos encontrados: {tamanhos}"
        )

    return [
        {
            "initPoints": ip,
            "iters.n": it,
            "acq": acq,
            "kernel": kernel,
        }
        for ip, it, acq, kernel in zip(
            pontos_iniciais,
            n_iteracoes,
            acqs,
            kernels,
            strict=True,
        )
    ]


def _build_base_combinacoes(cfg: Mapping[str, Any]) -> List[Dict[str, Any]]:
    pontos_iniciais = _parse_list(cfg.get("initPoints"), int)
    n_iteracoes = _parse_list(cfg.get("iters.n"), int)
    acqs = _parse_list(cfg.get("acq"), str)
    kernels = _parse_list(cfg.get("kernel"), str)

    tamanhos = {
        "initPoints": len(pontos_iniciais),
        "iters.n": len(n_iteracoes),
        "acq": len(acqs),
        "kernel": len(kernels),
    }

    if len(set(tamanhos.values())) != 1:
        raise ValueError(
            "As listas de configuração devem ter o mesmo tamanho. "
            f"Tamanhos encontrados: {tamanhos}"
        )

    return [
        {
            "initPoints": ip,
            "iters.n": it,
            "acq": acq,
            "kernel": kernel,
        }
        for ip, it, acq, kernel in zip(
            pontos_iniciais,
            n_iteracoes,
            acqs,
            kernels,
            strict=True,
        )
    ]


def _build_combinacoes(cfg: Mapping[str, Any]) -> List[Dict[str, Any]]:
    base_combinacoes = _build_base_combinacoes(cfg)
    bnx_pairs = _get_bnx_pairs(cfg)

    combinacoes: List[Dict[str, Any]] = []

    for pair in bnx_pairs:
        for comb in base_combinacoes:
            combinacoes.append(
                {
                    **comb,
                    "bnxFile": pair["bnxFile"],
                    "bnxTest": pair["bnxTest"],
                    "bnxLabel": pair["bnxLabel"],
                }
            )

    return combinacoes

        
def run_grid_calibrations(arq_config: Union[str, Path]) -> None:
    cfg = config_treatment(arq_config)
    input_list = build_input_list(cfg)

    base_combinacoes = _build_base_combinacoes(cfg)
    bnx_pairs = _get_bnx_pairs(cfg)

    for pair in bnx_pairs:
        bnx_file = pair["bnxFile"]
        bnx_test = pair["bnxTest"]
        bnx_label = pair["bnxLabel"]

        output_bnx_dir = Path("output") / bnx_label

        for base_comb in base_combinacoes:
            comb = {
                **base_comb,
                "bnxFile": bnx_file,
                "bnxTest": bnx_test,
                "bnxLabel": bnx_label,
            }

            run_simulation_baye(
                combinacao=comb,
                cfg=cfg,
                input_list=input_list,
                load_limites=load_limites,
                simulation_function=simulation_function,
                evaluate_difference=evaluate_difference,
                salvar_resultados_bo=salvar_resultados_bo,
                calcular_tempo_dec=calcular_tempo_dec,
                logfile="output/log_execucao.txt",
            )

        post_input = dict(input_list)
        post_input["bnxFile"] = bnx_file
        post_input["bnxTest"] = bnx_test
        post_input["bnxLabel"] = bnx_label

        run_post_training_pipeline_for_bnx(
            output_bnx_dir=output_bnx_dir,
            input_list=post_input,
            simulation_function=simulation_function,
            bnx_train=bnx_file,
            bnx_test=bnx_test,
        )
from __future__ import annotations

import os
import re
import shutil
from pathlib import Path
from typing import Any, List, Mapping, Union


BASE_FILES_REGEX = re.compile(
    r".*(\.ECO|\.SPE|\.WTH|\.X|\.SOL|\.CUL|\.T|\.A|\.ERR|\.CDE|\.L47|\.INP|\.INH|\.BNA|\.BNT|\.BNX|\.WDA)$",
    re.IGNORECASE,
)


def _copy_any(src: Path, dst_dir: Path) -> None:
    if src.is_dir():
        target = dst_dir / src.name
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(src, target)
    else:
        shutil.copy2(src, dst_dir / src.name)


def _copy_selected_bnx(input_list: Mapping[str, Any], sim_dir: Path) -> Path:
    bnx_dir = Path(str(input_list["bnxDir"]))
    bnx_file = str(input_list["bnxFile"])
    bnx_target_name = str(input_list.get("bnxTargetName", "EMEP1403.BNX"))

    src = bnx_dir / bnx_file
    dst = sim_dir / bnx_target_name

    if not src.exists():
        raise FileNotFoundError(f"Arquivo BNX não encontrado: {src}")

    shutil.copy2(src, dst)
    return dst


def files_dssat(
    dssat_file: Union[str, os.PathLike],
    experiment_directory: Union[str, os.PathLike],
    simulation_directory: Union[str, os.PathLike],
    input_list: Mapping[str, Any],
) -> List[str]:
    exp_dir = Path(experiment_directory)
    sim_dir = Path(simulation_directory)
    sim_dir.mkdir(parents=True, exist_ok=True)

    # copia arquivos base, exceto BNX
    experiment_files = [
        p
        for p in exp_dir.iterdir()
        if p.is_file()
        and BASE_FILES_REGEX.match(p.name)
        and p.suffix.upper() != ".BNX"
    ]

    for p in experiment_files:
        shutil.copy2(p, sim_dir / p.name)

    # copia executável
    _copy_any(Path(dssat_file), sim_dir)

    # copia BNX escolhido e renomeia para o nome esperado pelo DSSAT
    _copy_selected_bnx(input_list, sim_dir)

    sim_files = [str(p) for p in sim_dir.iterdir() if p.is_file() and BASE_FILES_REGEX.match(p.name)]
    return sim_files


def create_simulation_directories(
    param_sim: Mapping[str, Any],
    template_id: str,
    input_list: Mapping[str, Any],
) -> List[str]:
    dir_run = Path("runs")
    sim_dir = dir_run / template_id
    sim_dir.mkdir(parents=True, exist_ok=True)

    cultivar_file = Path(str(input_list["cultivarFile"]))
    cultivar = str(input_list["cultivar"])
    dssat_file = Path(str(input_list["dssatFile"]))
    experiment_dir = Path(str(input_list["dirExperiment"]))

    make_cultivar = input_list["make_cultivar"]
    write_cultivar = input_list["write_cultivar"]

    cultivar_df = make_cultivar(str(cultivar_file), dict(param_sim), cultivar)

    sim_files = files_dssat(
        str(dssat_file),
        str(experiment_dir),
        str(sim_dir),
        input_list,
    )

    write_cultivar(cultivar_df, str(sim_dir))

    return sim_files
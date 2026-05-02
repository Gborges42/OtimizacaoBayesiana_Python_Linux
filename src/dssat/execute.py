from __future__ import annotations

import os
import stat
import subprocess
import shutil
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Union

import pandas as pd

from src.dssat.io import pick_filex_name


def csm_batch(
    *,
    sim_dir: Path,
    filex_name: str,
    tn: Sequence[int],
    rp: int = 1,
    sq: int = 0,
    op: int = 0,
    co: int = 0,
    version: str = "47",
) -> Path:
    """
    Escreve DSSBatch.V47 (ou DSSBatch.v47) com 1 linha por tratamento.

    Formato baseado no padrão do DSSAT (FILEX com largura 92 e as colunas numéricas com largura 7),
    equivalente ao write_dssbatch() do pacote R DSSAT. :contentReference[oaicite:1]{index=1}
    """
    sim_dir = Path(sim_dir)
    sim_dir.mkdir(parents=True, exist_ok=True)

    # DSSAT costuma aceitar DSSBatch.V47; no seu baseline está DSSBatch.v47.
    # Para manter compatibilidade, use o mesmo nome que você já tem no projeto:
    batch_path = sim_dir / f"DSSBatch.v{version}"

    # Header (2 linhas): $BATCH + linha de colunas
    header_cols = f"{'@FILEX':<92}{'TRTNO':>7}{'RP':>7}{'SQ':>7}{'OP':>7}{'CO':>7}"
    lines = ["$BATCH", header_cols]

    # Uma linha por tratamento
    filex_field = f"{filex_name:<92}"
    for t in tn:
        lines.append(f"{filex_field}{int(t):>7}{int(rp):>7}{int(sq):>7}{int(op):>7}{int(co):>7}")

    batch_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        
    return batch_path


def execute_dssat(
    simulation_directory: Union[str, os.PathLike],
    dssat_file: Union[str, os.PathLike],
    model: str,
    batch_filename: str = "DSSBatch.v47",
    quiet: bool = True,
) -> bool:
    """
    Porta do R executeDssat():
      - roda ./<dssat_exec> <model> B DSSBatch.v47 dentro do diretório
      - valida se existem Evaluate.OUT e PlantGro.OUT
    """
    sim_dir = Path(simulation_directory)
    exe_name = Path(dssat_file).name
    exe_path = sim_dir / exe_name

    if not exe_path.exists():
        return False

    try:
        exe_path.chmod(exe_path.stat().st_mode | stat.S_IEXEC)
    except OSError:
        pass

    cmd = [f"./{exe_name}", str(model), "B", str(batch_filename)]

    subprocess.run(
        cmd,
        cwd=str(sim_dir),
        check=False,
        text=False,
        stdout=subprocess.DEVNULL if quiet else subprocess.PIPE,
        stderr=subprocess.DEVNULL if quiet else subprocess.PIPE,
    )

    return (sim_dir / "Evaluate.OUT").exists() and (sim_dir / "PlantGro.OUT").exists()


def run_dssat(
    simulation_files: Sequence[str],
    model: str,
    dssat_file: Union[str, os.PathLike],
    calibration: Sequence[str],
    read_treatments_id: Callable[[str], Sequence[float]],
    read_region: Callable[[str], str],
    read_evaluate: Callable[..., pd.DataFrame],
    cleanup: bool = True,
    crop: str = "BEAN",
    x_file: str | None = None,
) -> Optional[pd.DataFrame]:
    """
    Porta do R runDssat():
      - acha o diretório da simulação
      - acha o .X
      - lê treatmentId e region
      - cria DSSBatch.v47
      - executa DSSAT
      - lê Evaluate.OUT
      - limpa diretório
    """
    if not simulation_files:
        return None
    
    sim_dir = Path(simulation_files[0]).parent

    # acha X file
    x_files = [p for p in map(Path, simulation_files) if p.suffix.upper() == ".BNX"]
    if not x_files:
        return None
    x_file = str(x_files[0])

    region = read_region(x_file)

    # 1) descobrir qual BNX usar (ex.: EMEP1403.BNX)
    filex_name = pick_filex_name(sim_dir)
    # 2) ler lista de tratamentos do BNX (equivalente ao seu readTreatmentsId do R)
    tn_list = read_treatments_id(sim_dir / filex_name)
    # 3) reescrever o DSSBatch.v47 com o BNX correto + N linhas de TRTNO
    csm_batch(sim_dir=sim_dir, filex_name=filex_name, tn=tn_list, version="47")

    ok = execute_dssat(str(sim_dir), dssat_file, model, batch_filename="DSSBatch.v47", quiet=True)
    if not ok:
        if cleanup:
            shutil.rmtree(sim_dir, ignore_errors=True)
        return None

    df_eval = read_evaluate(str(sim_dir), region, calibration=calibration)

    if cleanup:
        shutil.rmtree(sim_dir, ignore_errors=True)

    return df_eval

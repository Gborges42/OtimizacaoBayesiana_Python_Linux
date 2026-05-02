from __future__ import annotations

import io
import os
import re
from pathlib import Path
from typing import Any, Optional, Sequence, Union

import numpy as np
import pandas as pd


def read_treatments_id(x_file: Union[str, os.PathLike]) -> np.ndarray:
    x_path = Path(x_file)
    x_lines = x_path.read_text(encoding="utf-8", errors="ignore").splitlines()

    label_idx = [i for i, ln in enumerate(x_lines) if re.search(r"\*[A-Z]", ln)]
    if not label_idx:
        raise ValueError("Não encontrei rótulos de grupo (linhas com '*[A-Z]') no arquivo .X")

    treat_positions = [j for j, idx in enumerate(label_idx) if "TREATMENTS" in x_lines[idx]]
    if not treat_positions:
        raise ValueError("Não encontrei o grupo 'TREATMENTS' no arquivo .X")

    pos = treat_positions[0]
    start = label_idx[pos] + 2
    if pos + 1 >= len(label_idx):
        raise ValueError("Não existe próximo rótulo após 'TREATMENTS' para definir intervalo.")
    end = label_idx[pos + 1] - 2

    if start > end:
        raise ValueError(f"Intervalo inválido no bloco TREATMENTS: start={start}, end={end}")

    block = x_lines[start : end + 1]

    ids = []
    for ln in block:
        toks = [t for t in ln.split(" ") if t != ""]

        if len(toks) >= 2:
            try:
                ids.append(float(toks[0]))
            except ValueError:
                continue

    return np.asarray(ids, dtype=float)


def read_region(x_file: Union[str, os.PathLike]) -> str:
    x_path = Path(x_file)
    x_lines = x_path.read_text(encoding="utf-8", errors="ignore").splitlines()

    for i, ln in enumerate(x_lines):
        if "@SITE" in ln:
            if i + 1 >= len(x_lines):
                raise ValueError("Encontrou '@SITE' mas não existe linha seguinte.")
            return x_lines[i + 1]
    raise ValueError("Não encontrei '@SITE' no arquivo .X")


def _day_of_year_from_yyyymmdd(date_val: Any) -> Optional[int]:
    if pd.isna(date_val):
        return None
    s = str(date_val).strip()
    # tenta YYYYMMDD
    if re.fullmatch(r"\d{8}", s):
        dt = pd.to_datetime(s, format="%Y%m%d", errors="coerce")
        if pd.isna(dt):
            return None
        return int(dt.dayofyear)
    dt = pd.to_datetime(s, errors="coerce")
    if pd.isna(dt):
        return None
    return int(dt.dayofyear)


def read_tfile(t_file: Union[str, os.PathLike]) -> pd.DataFrame:
    t_path = Path(t_file)
    lines = t_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    if not lines:
        raise ValueError("Arquivo .T vazio.")

    lines = [ln.replace("\t", "  ") for ln in lines]

    header_line = lines[0].replace("@", "")
    header = [h for h in header_line.split(" ") if h != ""]

    data_text = "\n".join(lines[1:])
    df = pd.read_csv(io.StringIO(data_text), sep=r'\s+', engine="python")

    if len(header) == df.shape[1]:
        df.columns = header
    else:
        min_len = min(len(header), df.shape[1])
        df = df.iloc[:, :min_len]
        df.columns = header[:min_len]

    if "DATE" in df.columns:
        df["DOY"] = df["DATE"].apply(_day_of_year_from_yyyymmdd)
    else:
        df["DOY"] = np.nan

    return df


def read_evaluate_og(
    simulation_directory: Union[str, os.PathLike],
    region: str,
    calibration: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    sim_dir = Path(simulation_directory)
    eval_file = sim_dir / "Evaluate.OUT"
    if not eval_file.exists():
        raise FileNotFoundError(f"Não encontrei Evaluate.OUT em {sim_dir}")

    df = pd.read_csv(eval_file, delim_whitespace=True, engine="python")
    df["region"] = region
    df["idRun"] = sim_dir.name

    return df

def read_evaluate(
    simulation_directory: Union[str, os.PathLike],
    region: str,
    calibration: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    sim_dir = Path(simulation_directory)
    eval_file = sim_dir / "Evaluate.OUT"
    if not eval_file.exists():
        raise FileNotFoundError(f"Não encontrei Evaluate.OUT em {sim_dir}")

    # 1) Encontra a linha de header (a que começa com '@')
    header_line = None
    with open(eval_file, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            s = line.strip()
            if s.startswith("@"):
                header_line = s
                break

    if header_line is None:
        raise ValueError(f"Não encontrei linha de cabeçalho iniciada por '@' em {eval_file}")

    # 2) Extrai nomes de colunas (removendo o '@' do primeiro token)
    # Ex.: "@RUN EXCODE TN ..." -> ["RUN", "EXCODE", "TN", ...]
    colnames = header_line.lstrip("@").split()

    # 3) Lê os dados: ignora comentários '*' e a linha '@' do cabeçalho
    df = pd.read_csv(
        eval_file,
        sep=r"\s+",
        engine="python",
        comment="*",          # ignora linhas que começam com '*'
        skiprows=1,           # pula a primeira linha (normalmente *EVALUATION...)
        names=colnames,       # força colunas corretas
        header=None,          # header vem de 'names'
    )

    # remove qualquer linha que por acaso tenha sobrado e comece com '@' (cabeçalho repetido)
    if "RUN" in df.columns:
        # RUN deveria ser numérico; se vier algo estranho, derruba
        df = df[pd.to_numeric(df["RUN"], errors="coerce").notna()].copy()

    df["region"] = region
    df["idRun"] = sim_dir.name

    return df


def read_plantgro_out(
    simulation_directory: Union[str, os.PathLike],
    treatment_id: Sequence[Union[int, float, str]],
) -> pd.DataFrame:
    sim_dir = Path(simulation_directory)
    pg_file = sim_dir / "PlantGro.OUT"
    if not pg_file.exists():
        raise FileNotFoundError(f"Não encontrei PlantGro.OUT em {sim_dir}")

    lines = pg_file.read_text(encoding="utf-8", errors="ignore").splitlines()

    cleaned = []
    for ln in lines:
        if "*" in ln:
            continue
        if "!" in ln:
            continue
        if ln.strip() == "":
            continue
        cleaned.append(ln)

    header_idx = [i for i, ln in enumerate(cleaned) if "@YEAR" in ln]
    if not header_idx:
        raise ValueError("Não encontrei cabeçalhos '@YEAR' em PlantGro.OUT")

    header_idx = header_idx + [len(cleaned)]
    treatments = [float(t) for t in treatment_id]

    blocks = []
    for block_i in range(len(header_idx) - 1):
        start = header_idx[block_i]
        end = header_idx[block_i + 1] - 1
        text = "\n".join(cleaned[start : end + 1])
        df = pd.read_csv(io.StringIO(text), delim_whitespace=True, engine="python")
        trno = treatments[block_i] if block_i < len(treatments) else np.nan
        df["TRNO"] = trno
        blocks.append(df)

    plantgro = pd.concat(blocks, ignore_index=True)
    plantgro.columns = [c.replace("@", "") for c in plantgro.columns]
    return plantgro

def pick_filex_name(sim_dir: Path, preferred: str | None = None) -> str:
    sim_dir = Path(sim_dir)

    # se o config já disser qual é o BNX, use
    if preferred:
        p = Path(preferred)
        return p.name

    # senão, procure o primeiro arquivo de experimento típico
    for ext in (".BNX", ".BMX", ".RIX", ".MZX", ".WHX"):  # ajuste se necessário
        found = sorted(sim_dir.glob(f"*{ext}"))
        if found:
            return found[0].name

    raise FileNotFoundError(f"Nenhum FILEX (.BNX/.BMX/...) encontrado em {sim_dir}")
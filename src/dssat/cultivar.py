from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Any, Dict, Mapping, Union

import numpy as np
import pandas as pd


def count_decimal_places(x: Any) -> int:
    s = str(x)
    if "." in s and ("e" not in s.lower()):
        return len(s.split(".", 1)[1])
    return 0


def _normalize_name(name: str) -> str:
    return str(name).replace("-", "")


def _row_to_mapping(multiply_vector: Any) -> Dict[str, float]:
    if isinstance(multiply_vector, dict):
        src = multiply_vector
    elif isinstance(multiply_vector, pd.Series):
        src = multiply_vector.to_dict()
    elif isinstance(multiply_vector, pd.DataFrame):
        if len(multiply_vector) == 0:
            return {}
        src = multiply_vector.iloc[0].to_dict()
    else:
        raise TypeError("multiply_vector deve ser dict, Series ou DataFrame(1 linha).")

    out: Dict[str, float] = {}
    for k, v in src.items():
        if v is None:
            continue
        try:
            fv = float(v)
        except (TypeError, ValueError):
            continue
        if np.isnan(fv):
            continue
        out[str(k)] = fv
    return out


def make_cultivar(
    cultivar_file: Union[str, os.PathLike],
    multiply_vector: Any,
    cultivar: str,
) -> pd.DataFrame:
    cultivar_path = Path(cultivar_file)
    lines = cultivar_path.read_text(encoding="utf-8", errors="ignore").splitlines()

    header_line = None
    for ln in lines:
        if "@VAR" in ln:
            header_line = ln
            break
    if header_line is None:
        raise ValueError("Não encontrei linha '@VAR' no arquivo .CUL.")

    cultivar_header = [tok for tok in header_line.split(" ") if tok != ""]

    cultivar_line = None
    for ln in lines:
        if cultivar in ln:
            cultivar_line = ln
            break
    if cultivar_line is None:
        raise ValueError(f"Não encontrei linha do cultivar '{cultivar}' no arquivo .CUL.")

    chars = list(cultivar_line)
    try:
        first_point_idx = chars.index(".")
    except ValueError:
        raise ValueError("Linha do cultivar não contém '.' para delimitar parte numérica.")

    post_point = "".join(chars[first_point_idx + 1 :])

    numeric_df = pd.read_csv(
        io.StringIO(post_point),
        sep=r"\s+",
        header=None,
        engine="python",
    )

    pre = pd.DataFrame([[cultivar, "RODADA_GA", "."]])
    cultivar_data = pd.concat([pre, numeric_df], axis=1)

    if len(cultivar_header) != cultivar_data.shape[1]:
        min_len = min(len(cultivar_header), cultivar_data.shape[1])
        cultivar_data = cultivar_data.iloc[:, :min_len]
        cultivar_header = cultivar_header[:min_len]

    cultivar_data.columns = cultivar_header

    mv = _row_to_mapping(multiply_vector)

    names_maior_norm = {_normalize_name(c): c for c in cultivar_data.columns}
    names_menor_norm = {_normalize_name(k): k for k in mv.keys()}
    comuns = set(names_maior_norm.keys()).intersection(set(names_menor_norm.keys()))

    for norm_name in comuns:
        col_dt = names_maior_norm[norm_name]
        col_mv = names_menor_norm[norm_name]

        original_value = cultivar_data.iloc[0][col_dt]
        dec = count_decimal_places(original_value)

        new_value = mv[col_mv]
        rounded_value = round(float(new_value), int(dec))
        cultivar_data.at[cultivar_data.index[0], col_dt] = rounded_value

    return cultivar_data


def write_cultivar(
    cultivar_data: pd.DataFrame,
    simulation_directory: Union[str, os.PathLike],
) -> None:
    if not isinstance(cultivar_data, pd.DataFrame):
        cultivar_data = pd.DataFrame(cultivar_data)

    if len(cultivar_data) == 0:
        raise ValueError("cultivar_data está vazio.")

    sim_dir = Path(simulation_directory)
    cul_files = sorted(sim_dir.glob("*.CUL"))
    if not cul_files:
        raise FileNotFoundError(f"Não encontrei arquivo .CUL em {sim_dir}")

    output_file = cul_files[0]
    cd = cultivar_data.copy()

    if cd.shape[1] >= 5:
        for col in cd.columns[4:]:
            s = str(cd.iloc[0][col])
            cd[col] = cd[col].astype(str)
            cd.at[cd.index[0], col] = s[:5]

    def concat(values) -> str:
        values = ["" if v is None else str(v) for v in values]
        parts = []
        for i, v in enumerate(values):
            if i == 0:
                parts.append(f"{v:<6s}")
            elif i == 1:
                parts.append(f"{v:<16s}")
            elif i == 2:
                parts.append(f"{v:>5s}")
            elif i == 3:
                parts.append(f"{v:>6s}")
            else:
                parts.append(f"{v:>5s}")
        return " ".join(parts)

    header = concat(list(cd.columns))
    line = concat(list(cd.iloc[0].tolist()))

    output_file.write_text(f"{header}\n{line}\n", encoding="utf-8")

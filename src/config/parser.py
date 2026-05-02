from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import json


def _parse_bool(x: str) -> Optional[bool]:
    s = x.strip().lower()
    if s in ("true", "t", "1", "yes", "y"):
        return True
    if s in ("false", "f", "0", "no", "n"):
        return False
    return None


def _parse_number(x: str) -> Any:
    s = x.strip()

    # int
    try:
        if s.isdigit() or (s.startswith("-") and s[1:].isdigit()):
            return int(s)
    except Exception:
        pass

    # float
    try:
        return float(s)
    except Exception:
        return s


def normalize_path_str(p: Any) -> str:
    """
    Normaliza caminhos como './/Baseline//Bean' -> 'Baseline/Bean' (no Linux)
    mantendo relativo ao projeto.
    """
    return str(Path(str(p)))


def _ensure_list(x: Any) -> List[Any]:
    if x is None:
        return []
    if isinstance(x, list):
        return x
    return [x]


def config_treatment(arq_config: Union[str, Path]) -> Dict[str, Any]:
    """
    Lê o arquivo .config (estilo do seu R) e retorna um dict TIPADO:

    - Ignora linhas vazias
    - Ignora linhas que contenham '!' (comentários)
    - Parseia 'key = value'
    - Se value tiver vírgulas, vira lista
    - Converte TRUE/FALSE -> bool
    - Converte números -> int/float quando possível
    - Mantém string quando não for número/bool

    Também normaliza alguns campos de path usados no pipeline:
      dirExperiment, dssatFile, cultivarFile, outputDir
    """
    path = Path(arq_config)
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()

    # remove vazios e comentários
    lines = [ln for ln in lines if ln.strip() and "!" not in ln]

    cfg: Dict[str, Any] = {}
    for ln in lines:
        if "=" not in ln:
            continue

        k, v = ln.split("=", 1)
        key = k.strip()
        raw = v.strip()

        parts = [p.strip() for p in raw.split(",")] if raw else []
        typed_parts: List[Any] = []

        for p in parts:
            b = _parse_bool(p)
            if b is not None:
                typed_parts.append(b)
            else:
                typed_parts.append(_parse_number(p))

        cfg[key] = typed_parts[0] if len(typed_parts) == 1 else typed_parts

    # normaliza paths mais importantes
    for k in ("dirExperiment", "dssatFile", "cultivarFile", "outputDir"):
        if k in cfg:
            cfg[k] = normalize_path_str(cfg[k])

    # garante que esses campos sejam listas (consistência com pipeline)
    for k in ("coefficients", "limites", "calibration"):
        if k in cfg:
            cfg[k] = _ensure_list(cfg[k])

    return cfg


def load_limites(cfg: Dict[str, Any]) -> Dict[str, List[float]]:
    """
    Lê cfg["limites"] e devolve dict compatível com BayesianOptimization.

    Seu .config:
      limites = EM-FL:15;55, FL-SH:2;10, ...

    Retorno:
      {"EMFL": [15.0, 55.0], "FLSH": [2.0, 10.0], ...}

    Observação: remove hífen no nome (igual ao R).
    """
    limites = cfg.get("limites", [])
    limites = _ensure_list(limites)

    out: Dict[str, List[float]] = {}
    for item in limites:
        s = str(item).strip()
        if not s:
            continue

        if ":" not in s or ";" not in s:
            raise ValueError(f"Limite inválido: '{s}'. Esperado 'NOME:low;high'.")

        name_part, range_part = s.split(":", 1)
        coef_name = name_part.strip().replace("-", "")

        low_str, high_str = range_part.split(";", 1)
        out[coef_name] = [float(low_str), float(high_str)]

    if not out:
        raise ValueError("Nenhum limite válido foi carregado de cfg['limites'].")
    return out

def _parse_list(value, cast=str):
    if isinstance(value, list):
        return [cast(v) for v in value]
    if isinstance(value, tuple):
        return [cast(v) for v in value]
    if isinstance(value, str):
        return [cast(v.strip()) for v in value.split(",") if v.strip()]
    return [cast(value)]

def attach_bnx_pairs(cfg: dict) -> dict:
    cfg = dict(cfg)

    bnx_files = _parse_list(cfg.get("bnxFiles", []), str)
    bnx_test = _parse_list(cfg.get("bnxTest", []), str)

    if not bnx_files:
        raise ValueError("bnxFiles não foi informado.")
    if not bnx_test:
        raise ValueError("bnxTest não foi informado.")
    if len(bnx_files) != len(bnx_test):
        raise ValueError(
            f"bnxFiles e bnxTest devem ter o mesmo tamanho. "
            f"Recebido: {len(bnx_files)} treino vs {len(bnx_test)} teste."
        )

    cfg["bnxFiles"] = bnx_files
    cfg["bnxTest"] = bnx_test
    cfg["bnxPairs"] = list(zip(bnx_files, bnx_test, strict=True))
    return cfg
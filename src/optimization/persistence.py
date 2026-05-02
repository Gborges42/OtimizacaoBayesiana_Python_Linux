from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Union, Optional

import pandas as pd
from joblib import dump
from joblib import load


def _get_score_summary(resultado_bo: Any) -> pd.DataFrame | None:
    rodadas = getattr(resultado_bo, "scoreSummary", None)
    if rodadas is None:
        rodadas = getattr(resultado_bo, "score_summary", None)
    if rodadas is None:
        return None
    if isinstance(rodadas, pd.DataFrame):
        return rodadas
    return pd.DataFrame(rodadas)


def _build_light_payload(resultado_bo: Any) -> Dict[str, Any]:
    payload: Dict[str, Any] = {}
    payload["best_params"] = getattr(resultado_bo, "best_params", None)
    payload["best_score"] = getattr(resultado_bo, "best_score", None)
    payload["config"] = getattr(resultado_bo, "config", None)

    score_summary = _get_score_summary(resultado_bo)
    payload["scoreSummary"] = score_summary

    opt = getattr(resultado_bo, "optimizer", None)
    if opt is not None:
        payload["optimizer_max"] = getattr(opt, "max", None)
        payload["optimizer_res"] = getattr(opt, "res", None)

    return payload


def salvar_resultados_bo(
    resultado_bo: Any,
    caminho_output: Union[str, os.PathLike],
    valores_sufixo: List[str],
) -> None:
    """
    Salva resultados do BO:
      - joblib: tenta salvar o objeto completo; se falhar, salva um payload leve
      - CSV: todas rodadas
      - CSV: melhor rodada com metadados da configuração
    """
    out_dir = Path(caminho_output)
    out_dir.mkdir(parents=True, exist_ok=True)

    method = str(valores_sufixo[0])
    initp = str(valores_sufixo[1])
    itersn = str(valores_sufixo[2])
    acq = str(valores_sufixo[3])
    kernel = str(valores_sufixo[4])

    #joblib_path = out_dir / f"bayeopt_{initp}_{itersn}_{acq}_{kernel}.joblib"

    #try:
    #    dump(resultado_bo, joblib_path)
    #except Exception as e:
    #    light = _build_light_payload(resultado_bo)
    #    joblib_path_light = out_dir / f"bayeopt_{initp}_{itersn}_{acq}_{kernel}.light.joblib"
    #    dump(light, joblib_path_light)

    #    err_path = out_dir / f"bayeopt_{initp}_{itersn}_{acq}_{kernel}.pickle_error.txt"
    #    err_path.write_text(str(e), encoding="utf-8")

    rodadas = _get_score_summary(resultado_bo)
    if rodadas is None:
        return

    rodadas_path = out_dir / f"todas_rodadas_{initp}_{itersn}_{acq}_{kernel}.csv"
    rodadas.to_csv(rodadas_path, index=False)

    if "Score" in rodadas.columns and len(rodadas) > 0:
        melhor = rodadas.loc[rodadas["Score"].astype(float).idxmax()].to_frame().T

        # metadados da configuração usados depois no pós-treinamento
        melhor["initPoints"] = initp
        melhor["iters.n"] = itersn
        melhor["acq"] = acq
        melhor["kernel"] = kernel
        melhor["method"] = method

        melhor_path = out_dir / f"melhor_resultado_{initp}_{itersn}_{acq}_{kernel}.csv"
        melhor.to_csv(melhor_path, index=False)


def carregar_resultados_bo(
    caminho_output: Union[str, os.PathLike],
    *,
    method: Optional[str] = None,
    initp: Optional[Union[int, str]] = None,
    itersn: Optional[Union[int, str]] = None,
    path: Optional[Union[str, os.PathLike]] = None,
) -> Dict[str, Any]:
    """
    Carrega resultados do BO de forma compatível com:
      - bayeopt_{method}_{initp}_{itersn}.joblib  (objeto completo)
      - bayeopt_{method}_{initp}_{itersn}.light.joblib (payload leve)

    Formas de uso:
      1) Passando path direto:
         load_resultados_bo("output", path="output/bayeopt_RMSE_10_5.light.joblib")

      2) Passando method/initp/itersn (ele encontra o arquivo):
         load_resultados_bo("output", method="RMSE", initp=10, itersn=5)

    Retorna um dict "normalizado" com chaves:
      - raw: objeto carregado (seja completo ou payload)
      - is_light: bool
      - scoreSummary: pd.DataFrame | None
      - best_params: dict | None
      - best_score: float | None
      - config: dict | None
      - optimizer_max: Any | None
      - optimizer_res: Any | None
      - source_path: str (arquivo carregado)
    """
    out_dir = Path(caminho_output)

    if path is not None:
        chosen = Path(path)
        if not chosen.is_absolute():
            chosen = out_dir / chosen
        if not chosen.exists():
            raise FileNotFoundError(f"Arquivo não encontrado: {chosen}")
    else:
        if method is None or initp is None or itersn is None:
            raise ValueError("Informe (method, initp, itersn) ou um path explícito.")

        method_s = str(method)
        initp_s = str(initp)
        itersn_s = str(itersn)

        # prioridade: full -> light
        full = out_dir / f"bayeopt_{method_s}_{initp_s}_{itersn_s}.joblib"
        light = out_dir / f"bayeopt_{method_s}_{initp_s}_{itersn_s}.light.joblib"

        if full.exists():
            chosen = full
        elif light.exists():
            chosen = light
        else:
            raise FileNotFoundError(
                "Não encontrei arquivo de resultado.\n"
                f"Tentei:\n- {full}\n- {light}"
            )

    obj = load(chosen)

    # --- normalização ---
    is_light = isinstance(obj, dict) and ("scoreSummary" in obj or "optimizer_max" in obj)

    def _to_df(x: Any) -> Optional[pd.DataFrame]:
        if x is None:
            return None
        if isinstance(x, pd.DataFrame):
            return x
        try:
            return pd.DataFrame(x)
        except Exception:
            return None

    if is_light:
        score_summary = _to_df(obj.get("scoreSummary"))
        best_params = obj.get("best_params")
        best_score = obj.get("best_score")
        config = obj.get("config")
        optimizer_max = obj.get("optimizer_max")
        optimizer_res = obj.get("optimizer_res")
    else:
        # objeto completo (ex.: BayeOptResult)
        score_summary = _to_df(getattr(obj, "scoreSummary", None) or getattr(obj, "score_summary", None))
        best_params = getattr(obj, "best_params", None)
        best_score = getattr(obj, "best_score", None)
        config = getattr(obj, "config", None)

        opt = getattr(obj, "optimizer", None)
        optimizer_max = getattr(opt, "max", None) if opt is not None else None
        optimizer_res = getattr(opt, "res", None) if opt is not None else None

    return {
        "raw": obj,
        "is_light": is_light,
        "scoreSummary": score_summary,
        "best_params": best_params,
        "best_score": best_score,
        "config": config,
        "optimizer_max": optimizer_max,
        "optimizer_res": optimizer_res,
        "source_path": str(chosen),
    }
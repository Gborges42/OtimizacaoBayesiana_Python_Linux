from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

EPS = 1e-12


# =========================================================
# Utilidades de colunas
# =========================================================
def _norm_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _find_col(df: pd.DataFrame, candidates: Sequence[str]) -> str | None:
    cols_lower = {str(c).strip().lower(): c for c in df.columns}
    for cand in candidates:
        key = str(cand).strip().lower()
        if key in cols_lower:
            return cols_lower[key]

    for cand in candidates:
        for c in df.columns:
            if str(cand).strip().lower() == str(c).strip().lower():
                return c
    return None


def _require_cols(df: pd.DataFrame, cols: Sequence[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Colunas ausentes: {missing}\n"
            f"Colunas disponíveis: {list(df.columns)}"
        )


# =========================================================
# Métricas reaproveitadas do comparar_simulacoes.py
# =========================================================
def relative_error(sim: np.ndarray, meas: np.ndarray) -> np.ndarray:
    sim = np.asarray(sim, dtype=float)
    meas = np.asarray(meas, dtype=float)

    ok = np.isfinite(sim) & np.isfinite(meas) & (np.abs(meas) > EPS)
    out = np.full(sim.shape, np.nan, dtype=float)
    out[ok] = (sim[ok] - meas[ok]) / meas[ok]
    return out


def clean_2d(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    ok = np.all(np.isfinite(X), axis=1)
    return X[ok]


def metric_euclidean_2d(E: np.ndarray) -> float:
    E = clean_2d(E)
    if E.shape[0] == 0:
        return np.nan
    norms = np.sqrt(np.sum(E ** 2, axis=1))
    return float(np.mean(norms))


# =========================================================
# Leitura dos CSVs de "melhor_resultado"
# =========================================================
def load_best_result_csvs(output_bnx_dir: str | Path) -> pd.DataFrame:
    out_dir = Path(output_bnx_dir)
    files = sorted(out_dir.glob("melhor_resultado_*.csv"))

    if not files:
        raise FileNotFoundError(
            f"Nenhum arquivo melhor_resultado_*.csv encontrado em {out_dir}"
        )

    frames: List[pd.DataFrame] = []
    for f in files:
        df = pd.read_csv(f)
        df["source_file"] = f.name
        frames.append(df)

    return pd.concat(frames, ignore_index=True)


def _detect_config_columns(df: pd.DataFrame) -> Tuple[str, str, str, str]:
    cols = {str(c).strip().lower(): c for c in df.columns}

    init_col = cols.get("initpoints")
    iters_col = cols.get("iters.n") or cols.get("iters_n")
    acq_col = cols.get("acq")
    kernel_col = cols.get("kernel")

    if not all([init_col, iters_col, acq_col, kernel_col]):
        raise ValueError(
            "Não encontrei colunas de configuração esperadas "
            "('initPoints', 'iters.n', 'acq', 'kernel') nos melhores resultados.\n"
            f"Colunas disponíveis: {list(df.columns)}"
        )

    return init_col, iters_col, acq_col, kernel_col


def extract_best_runs_table(
    best_results_df: pd.DataFrame,
    *,
    bnx_train: str,
    bnx_test: str,
) -> pd.DataFrame:
    df = best_results_df.copy()

    if "Score" not in df.columns:
        raise ValueError(
            f"Não encontrei a coluna 'Score' em melhor_resultado. Colunas: {list(df.columns)}"
        )

    init_col, iters_col, acq_col, kernel_col = _detect_config_columns(df)

    reserved_cols = {
        "source_file",
        "Score",
        init_col,
        iters_col,
        acq_col,
        kernel_col,
        "method",
        "bnx_label",
        "bnx_train",
        "bnx_test",
    }

    param_cols = [c for c in df.columns if c not in reserved_cols]

    rows: List[Dict[str, Any]] = []

    for _, row in df.iterrows():
        label = f"IP{row[init_col]}_IT{row[iters_col]}_{row[acq_col]}_{row[kernel_col]}"

        best_params: Dict[str, float] = {}
        for p in param_cols:
            try:
                best_params[p] = float(row[p])
            except Exception:
                continue

        rows.append(
            {
                "label": label,
                "initPoints": str(row[init_col]),
                "iters_n": str(row[iters_col]),
                "acq": str(row[acq_col]),
                "kernel": str(row[kernel_col]),
                "best_score": float(row["Score"]),
                "best_params": best_params,
                "bnx_train": bnx_train,
                "bnx_test": bnx_test,
            }
        )

    return pd.DataFrame(rows)


# =========================================================
# Cálculo do euclidiano a partir do Evaluate
# =========================================================
def calculate_euclidean_summary(
    evaluate_df: pd.DataFrame,
    *,
    label_col: str = "simulacao",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Retorna:
      - per_run: uma linha por (simulacao, RUN)
      - per_sim: média por simulacao
    """
    df = _norm_cols(evaluate_df)

    run_col = _find_col(df, ["@RUN", "RUN"])
    simcfg_col = _find_col(df, [label_col, "simulação", "simulation", "config"])

    if run_col is None:
        raise ValueError("Não encontrei coluna RUN/@RUN no Evaluate.")
    if simcfg_col is None:
        raise ValueError(
            f"Não encontrei coluna de simulação ('{label_col}', 'simulação', 'simulation', 'config')."
        )

    pairs = {
        "ADAP": ("ADAPS", "ADAPM"),
        "PD1P": ("PD1PS", "PD1PM"),
        "PDFP": ("PDFPS", "PDFPM"),
        "MDAP": ("MDAPS", "MDAPM"),
        "HWAM": ("HWAMS", "HWAMM"),
        "PWAM": ("PWAMS", "PWAMM"),
        "HWUM": ("HWUMS", "HWUMM"),
        "H#AM": ("H#AMS", "H#AMM"),
        "H#UM": ("H#UMS", "H#UMM"),
        "THAM": ("THAMS", "THAMM"),
        "LAIX": ("LAIXS", "LAIXM"),
        "HIAM": ("HIAMS", "HIAMM"),
    }

    needed: List[str] = [run_col, simcfg_col]
    for sim_col, meas_col in pairs.values():
        needed.extend([sim_col, meas_col])
    _require_cols(df, needed)

    for v, (sim_col, meas_col) in pairs.items():
        df[f"e_{v}"] = relative_error(df[sim_col].to_numpy(), df[meas_col].to_numpy())

    rows: List[Dict[str, Any]] = []

    grouped = df.groupby([simcfg_col, run_col], dropna=False)
    for (simcfg, run), g in grouped:
        E = g[[f"e_{v}" for v in pairs.keys()]].to_numpy()
        E = clean_2d(E)

        out: Dict[str, Any] = {
            "simulacao": simcfg,
            "run": run,
            "EUCLIDIANA_GERAL": metric_euclidean_2d(E),
        }

        for v in pairs.keys():
            e = g[f"e_{v}"].to_numpy()
            e = e[np.isfinite(e)]
            out[f"EUCLIDIANA_{v}"] = float(np.mean(np.abs(e))) if e.size > 0 else np.nan

        rows.append(out)

    per_run = pd.DataFrame(rows)
    per_sim = per_run.groupby("simulacao", dropna=False).mean(numeric_only=True).reset_index()

    return per_run, per_sim


def save_euclidean_plot(
    euclid_df: pd.DataFrame,
    outpath: str | Path,
    *,
    title: str,
) -> None:
    if euclid_df.empty:
        return
    if "simulacao" not in euclid_df.columns or "EUCLIDIANA_GERAL" not in euclid_df.columns:
        return

    df = euclid_df.copy().sort_values("EUCLIDIANA_GERAL", ascending=True)

    fig_h = max(4, 0.55 * len(df))
    plt.figure(figsize=(11, fig_h))
    plt.barh(df["simulacao"].astype(str), df["EUCLIDIANA_GERAL"].astype(float))
    plt.gca().invert_yaxis()
    plt.xlabel("Euclidiana geral (menor = melhor)")
    plt.ylabel("Simulação")
    plt.title(title)
    plt.tight_layout()

    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath, dpi=220, bbox_inches="tight")
    plt.close()


# =========================================================
# Execução singular reaproveitando simulation_function()
# =========================================================
def run_single_dssat_simulation(
    *,
    best_params: Mapping[str, float],
    label: str,
    input_list: Mapping[str, Any],
    simulation_function: Callable[[Mapping[str, Any], str, Mapping[str, Any]], pd.DataFrame],
    template_prefix: str,
) -> pd.DataFrame:
    df = simulation_function(
        best_params,
        f"{template_prefix}_{label}",
        dict(input_list),
    )
    return df if isinstance(df, pd.DataFrame) else pd.DataFrame()


# =========================================================
# Pipeline pós-treinamento por BNX
# =========================================================
def run_post_training_pipeline_for_bnx(
    *,
    output_bnx_dir: str | Path,
    input_list: Mapping[str, Any],
    simulation_function: Callable[[Mapping[str, Any], str, Mapping[str, Any]], pd.DataFrame],
    bnx_train: str,
    bnx_test: str,
) -> Dict[str, Any]:
    """
    Fluxo:
      1) lê os melhores_resultado_* do BNX de treino
      2) roda DSSAT singular para cada melhor configuração com BNX de treino
      3) calcula e salva euclidiano de treinamento
      4) roda DSSAT singular para as MESMAS configurações com BNX de teste
      5) calcula e salva euclidiano de teste
      6) escolhe a melhor configuração final pelo menor EUCLIDIANA_GERAL do teste
      7) salva melhor_configuracao_pos_treinamento.csv
    """
    out_dir = Path(output_bnx_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    best_results_df = load_best_result_csvs(out_dir)
    runs_df = extract_best_runs_table(
        best_results_df,
        bnx_train=bnx_train,
        bnx_test=bnx_test,
    )

    # -------------------------
    # Treinamento
    # -------------------------
    train_eval_frames: List[pd.DataFrame] = []

    for _, row in runs_df.iterrows():
        il_train = dict(input_list)
        il_train["bnxFile"] = row["bnx_train"]

        eval_df = run_single_dssat_simulation(
            best_params=row["best_params"],
            label=row["label"],
            input_list=il_train,
            simulation_function=simulation_function,
            template_prefix="treino",
        )

        if eval_df is None or eval_df.empty:
            continue

        eval_df = eval_df.copy()
        eval_df["simulacao"] = row["label"]
        train_eval_frames.append(eval_df)

    if not train_eval_frames:
        raise RuntimeError(
            f"Nenhum Evaluate válido foi gerado para o pós-treinamento de {bnx_train}"
        )

    evaluate_train = pd.concat(train_eval_frames, ignore_index=True)
    train_per_run, train_per_sim = calculate_euclidean_summary(
        evaluate_train,
        label_col="simulacao",
    )

    evaluate_train.to_csv(out_dir / "evaluate_treinamento_best.csv", index=False, encoding="utf-8")
    train_per_run.to_csv(out_dir / "euclidiano_treinamento_por_run.csv", index=False, encoding="utf-8")
    train_per_sim.to_csv(out_dir / "euclidiano_treinamento.csv", index=False, encoding="utf-8")

    save_euclidean_plot(
        train_per_sim,
        out_dir / "grafico_euclidiano_treinamento.png",
        title=f"Treinamento — {Path(bnx_train).stem}",
    )

    if train_per_sim.empty or "EUCLIDIANA_GERAL" not in train_per_sim.columns:
        raise RuntimeError(
            f"Não foi possível calcular o euclidiano de treinamento para {bnx_train}"
        )
        
    # Melhor configuração no TREINAMENTO
    best_train_row = train_per_sim.sort_values("EUCLIDIANA_GERAL", ascending=True).iloc[0]
    best_train_label = str(best_train_row["simulacao"])

    best_train_run = runs_df.loc[runs_df["label"] == best_train_label]
    if best_train_run.empty:
        raise RuntimeError(
            f"Não encontrei a configuração de treino '{best_train_label}' em runs_df"
        )
    best_train_run = best_train_run.iloc[0]

    best_train_info = pd.DataFrame(
        [
            {
                "simulacao_escolhida_treinamento": best_train_run["label"],
                "bnx_train": best_train_run["bnx_train"],
                "euclidiana_treinamento": float(best_train_row["EUCLIDIANA_GERAL"]),
                **best_train_run["best_params"],
            }
        ]
    )

    best_train_info.to_csv(
        out_dir / "melhor_configuracao_treinamento.csv",
        index=False,
        encoding="utf-8",
    )

    # -------------------------
    # Teste
    # -------------------------
    # IMPORTANTE:
    # roda o teste para TODAS as mesmas melhores configurações do treino,
    # e só depois escolhe a melhor pelo menor euclidiano do teste.
    test_eval_frames: List[pd.DataFrame] = []

    for _, row in runs_df.iterrows():
        il_test = dict(input_list)
        il_test["bnxFile"] = row["bnx_test"]

        eval_df = run_single_dssat_simulation(
            best_params=row["best_params"],
            label=f"{row['label']}_teste",
            input_list=il_test,
            simulation_function=simulation_function,
            template_prefix="teste",
        )

        if eval_df is None or eval_df.empty:
            continue

        eval_df = eval_df.copy()
        eval_df["simulacao"] = row["label"]
        test_eval_frames.append(eval_df)

    if not test_eval_frames:
        raise RuntimeError(
            f"Nenhum Evaluate válido foi gerado para o teste de {bnx_test}"
        )

    evaluate_test = pd.concat(test_eval_frames, ignore_index=True)
    test_per_run, test_per_sim = calculate_euclidean_summary(
        evaluate_test,
        label_col="simulacao",
    )

    evaluate_test.to_csv(out_dir / "evaluate_teste_best.csv", index=False, encoding="utf-8")
    test_per_run.to_csv(out_dir / "euclidiano_teste_por_run.csv", index=False, encoding="utf-8")
    test_per_sim.to_csv(out_dir / "euclidiano_teste.csv", index=False, encoding="utf-8")

    save_euclidean_plot(
        test_per_sim,
        out_dir / "grafico_euclidiano_teste.png",
        title=f"Teste — {Path(bnx_test).stem}",
    )

    if test_per_sim.empty or "EUCLIDIANA_GERAL" not in test_per_sim.columns:
        raise RuntimeError(
            f"Não foi possível calcular o euclidiano de teste para {bnx_test}"
        )

    # Seleção final agora é feita pelo TESTE
    selected_test = test_per_sim.sort_values("EUCLIDIANA_GERAL", ascending=True).iloc[0]
    selected_label = str(selected_test["simulacao"])

    selected_run = runs_df.loc[runs_df["label"] == selected_label]
    if selected_run.empty:
        raise RuntimeError(
            f"Não encontrei a configuração selecionada '{selected_label}' em runs_df"
        )
    selected_run = selected_run.iloc[0]

    # Captura também a euclidiana correspondente do treino
    selected_train = train_per_sim.loc[train_per_sim["simulacao"] == selected_label]
    train_best_euclid = np.nan
    if not selected_train.empty and "EUCLIDIANA_GERAL" in selected_train.columns:
        train_best_euclid = float(selected_train.iloc[0]["EUCLIDIANA_GERAL"])

    test_best_euclid = float(selected_test["EUCLIDIANA_GERAL"])

    selected_info = pd.DataFrame(
        [
            {
                "simulacao_escolhida_final": selected_run["label"],
                "bnx_train": selected_run["bnx_train"],
                "bnx_test": selected_run["bnx_test"],
                "euclidiana_treinamento": train_best_euclid,
                "euclidiana_teste": test_best_euclid,
                **selected_run["best_params"],
            }
        ]
    )
    selected_info.to_csv(
        out_dir / "melhor_configuracao_pos_treinamento.csv",
        index=False,
        encoding="utf-8",
    )

    return {
        "train_per_run": train_per_run,
        "train_per_sim": train_per_sim,
        "evaluate_train": evaluate_train,
        "test_per_run": test_per_run,
        "test_per_sim": test_per_sim,
        "evaluate_test": evaluate_test,
        "selected_label": selected_run["label"],
        "selected_bnx_train": selected_run["bnx_train"],
        "selected_bnx_test": selected_run["bnx_test"],
    }
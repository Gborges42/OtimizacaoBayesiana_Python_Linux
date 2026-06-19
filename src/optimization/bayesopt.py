from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
from bayes_opt import BayesianOptimization, UtilityFunction
from joblib import Parallel, delayed

from src.utils.logging import append_log
from src.scoring.objective import scoring_function

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, ExpSineSquared, ConstantKernel as C

@dataclass
class BayeOptResult:
    optimizer: BayesianOptimization
    scoreSummary: pd.DataFrame
    best_params: Dict[str, float]
    best_score: float
    config: Dict[str, Any]


def _seed_to_random_state(seed: int) -> int:
    """
    seed=0 => seed aleatória (como seu comentário no .config).
    """
    if int(seed) == 0:
        return int(np.random.randint(1, 2_000_000_000))
    return int(seed)


# Define um modelo customizado de BayesianOptimization
class CustomBayesianOptimization(BayesianOptimization):
    def __init__(self, f, pbounds, kernel=None, *args, **kwargs):
        # Cria um GaussianProcessRegressor com o kernel fornecido
        self.kernel = kernel if kernel else RBF(length_scale=1.0)  # Defina o kernel padrão aqui
        self.gpr = GaussianProcessRegressor(kernel=self.kernel)
        
        # Inicializa a classe pai
        super().__init__(f, pbounds, *args, **kwargs)

    def _initialize_model(self):
        # Substitua o modelo interno pelo modelo customizado (com o kernel)
        self.model = self.gpr

def run_simulation_baye(
    combinacao: Mapping[str, Any],
    *,
    cfg: Mapping[str, Any],
    input_list: Mapping[str, Any],
    load_limites: Callable[[Mapping[str, Any]], Dict[str, List[float]]],
    simulation_function: Callable[[Mapping[str, Any], str, Mapping[str, Any]], pd.DataFrame],
    evaluate_difference: Callable[[pd.DataFrame, Iterable[str], str, float], float],
    salvar_resultados_bo: Callable[[Any, str, List[str]], Any],
    calcular_tempo_dec: Callable[[datetime], str],
    logfile: str = "output/log_execucao.txt",
) -> BayeOptResult:
    # bounds
    bounds_raw = load_limites(dict(cfg))
    pbounds: Dict[str, Tuple[float, float]] = {
        k: (float(v[0]), float(v[1])) for k, v in bounds_raw.items()
    }

    # parâmetros da rodada
    init_points = int(combinacao.get("initPoints", cfg.get("initPoints", 10)))
    iters_n = int(combinacao.get("iters.n", cfg.get("iters.n", 5)))
    iters_k = int(cfg.get("iters.k", 1))

    acq = str(combinacao.get("acq", cfg.get("acq", "ucb"))).lower()
    kernel_type = str(combinacao.get("kernel", cfg.get("kernel", "gauss"))).lower()
    bnx_file = str(combinacao.get("bnxFile", cfg.get("bnxFile", ""))).strip()
    bnx_label = str(combinacao.get("bnxLabel", Path(bnx_file).stem if bnx_file else "SEM_BNX"))

    paralelo = bool(cfg.get("paralelo", False))
    cores = max(int(cfg.get("cores", 1)), 1)

    # kernel
    if kernel_type == "gauss":
        kernel = RBF(length_scale=1.0)
    elif kernel_type == "matern3_2":
        kernel = Matern(length_scale=1.0, nu=1.5)
    elif kernel_type == "matern5_2":
        kernel = Matern(length_scale=1.0, nu=2.5)
    elif kernel_type == "exp":
        kernel = ExpSineSquared(length_scale=1.0, periodicity=3.0)
    else:
        raise ValueError(
            f"Kernel '{kernel_type}' não reconhecido. "
            "Use 'gauss', 'matern3_2', 'matern5_2' ou 'exp'."
        )

    metodo_score = str(cfg.get("metodo_score", "rmse")).upper()
    output_dir = str(cfg.get("outputDir", "output"))
    output_dir = str(Path(output_dir) / bnx_label)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # log início
    start_time = datetime.now()
    append_log(
        logfile,
        "*******************************************************\n"
        f"** Iniciando o processo de otimização com {metodo_score}. "
        f"Tempo: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"** Rodada: initPoints={init_points}, iters.n={iters_n}, acq={acq}, kernel={kernel_type}, bnx={bnx_file}\n"
    )

    random_state = _seed_to_random_state(int(cfg.get("seed", 0)))

    rodada_input_list = dict(input_list)
    rodada_input_list["bnxFile"] = bnx_file
    rodada_input_list["bnxLabel"] = bnx_label
    rodada_input_list["metodo_score"] = str(cfg.get("metodo_score", "rmse")).lower()
    rodada_input_list["mape_iqr_lambda"] = float(cfg.get("mape_iqr_lambda", 1.5))

    # função objetivo
    def _objective(**kwargs: float) -> float:
        il = dict(rodada_input_list)

        try:
            return scoring_function(
                kwargs,
                il,
                simulation_function=simulation_function,
                evaluate_difference=evaluate_difference,
                logfile=logfile,
                noise_std=1e-4,
            )
        except Exception as e:
            append_log(logfile, f"[BO] objective exception: {repr(e)} params={kwargs}\n")
            return -99.0

    # optimizer
    optimizer = CustomBayesianOptimization(
        f=_objective,
        pbounds=pbounds,
        kernel=kernel,
        random_state=random_state,
        verbose=2,
        allow_duplicate_points=True,  # rede de segurança
    )

    # utility / acquisition
    if acq == "ucb":
        utility = UtilityFunction(kind="ucb", kappa=float(cfg.get("kappa", 2.576)), xi=float(cfg.get("xi", 0.0)))
    elif acq == "ei":
        utility = UtilityFunction(kind="ei", xi=float(cfg.get("xi", 0.01)))
    elif acq == "poi":
        utility = UtilityFunction(kind="poi", xi=float(cfg.get("xi", 0.01)))
    else:
        raise ValueError(f"Função de aquisição '{acq}' não reconhecida. Use 'ucb', 'ei' ou 'poi'.")

    history: List[Dict[str, Any]] = []
    keys = list(pbounds.keys())
    rng = np.random.default_rng(random_state)

    def _point_signature(params: Dict[str, float], ndigits: int = 10) -> tuple:
        """
        Assinatura hashable do ponto. O arredondamento evita problemas por
        diferenças mínimas de ponto flutuante.
        """
        return tuple(round(float(params[k]), ndigits) for k in keys)

    pontos_registrados = set()

    def _registrar_se_unico(params: Dict[str, float], target: float) -> bool:
        assinatura = _point_signature(params)
        if assinatura in pontos_registrados:
            append_log(logfile, f"[BO] ponto duplicado ignorado: {params}\n")
            return False

        optimizer.register(params=params, target=float(target))
        pontos_registrados.add(assinatura)
        return True

    def _sample_from_pbounds() -> Dict[str, float]:
        params: Dict[str, float] = {}
        for k, (lo, hi) in pbounds.items():
            if lo == hi:
                params[k] = float(lo)
            else:
                params[k] = float(rng.uniform(lo, hi))
        return params

    # ---------------------------
    # pontos iniciais únicos
    # ---------------------------
    init_params: List[Dict[str, float]] = []
    assinaturas_init = set()
    max_tentativas_init = max(init_points * 20, 100)

    while len(init_params) < init_points and max_tentativas_init > 0:
        candidato = _sample_from_pbounds()
        assinatura = _point_signature(candidato)

        if assinatura not in assinaturas_init and assinatura not in pontos_registrados:
            init_params.append(candidato)
            assinaturas_init.add(assinatura)

        max_tentativas_init -= 1

    if len(init_params) < init_points:
        append_log(
            logfile,
            f"[BO] aviso: só foi possível gerar {len(init_params)} pontos iniciais únicos "
            f"de {init_points} solicitados.\n"
        )

    if paralelo and cores > 1 and len(init_params) > 1:
        init_scores = Parallel(n_jobs=cores)(
            delayed(_objective)(**d) for d in init_params
        )
    else:
        init_scores = [_objective(**d) for d in init_params]

    for d, s in zip(init_params, init_scores):
        if _registrar_se_unico(d, float(s)):
            history.append({"Score": float(s), **d})

    if optimizer.max is None:
        raise RuntimeError(
            "Nenhum ponto inicial foi registrado no BayesianOptimization. "
            "Verifique se pbounds está correto e se a função objetivo está retornando valores numéricos."
        )

    # ---------------------------
    # passos do BO em batches
    # ---------------------------
    total_steps = iters_n * max(iters_k, 1)
    done = 0

    while done < total_steps:
        batch = min(max(iters_k, 1), total_steps - done)

        suggestions: List[Dict[str, float]] = []
        assinaturas_lote = set()
        tentativas = 0
        max_tentativas_suggest = batch * 20

        while len(suggestions) < batch and tentativas < max_tentativas_suggest:
            s = optimizer.suggest(utility)
            assinatura = _point_signature(s)

            if assinatura not in pontos_registrados and assinatura not in assinaturas_lote:
                suggestions.append(s)
                assinaturas_lote.add(assinatura)

            tentativas += 1

        if not suggestions:
            append_log(
                logfile,
                "[BO] nenhuma sugestão única encontrada nesta iteração; encerrando antecipadamente.\n"
            )
            break

        if paralelo and cores > 1 and len(suggestions) > 1:
            scores = Parallel(n_jobs=cores)(
                delayed(_objective)(**s) for s in suggestions
            )
        else:
            scores = [_objective(**s) for s in suggestions]

        registrados_no_batch = 0
        for s, sc in zip(suggestions, scores):
            if _registrar_se_unico(s, float(sc)):
                history.append({"Score": float(sc), **s})
                registrados_no_batch += 1

        if registrados_no_batch == 0:
            append_log(
                logfile,
                "[BO] nenhuma sugestão nova foi registrada neste batch; encerrando antecipadamente.\n"
            )
            break

        done += len(suggestions)

    if not history:
        raise RuntimeError("Nenhum resultado foi produzido pela otimização bayesiana.")

    score_summary = pd.DataFrame(history)

    best_idx = int(score_summary["Score"].astype(float).idxmax())
    best_row = score_summary.loc[best_idx]
    best_score = float(best_row["Score"])
    best_params = {k: float(best_row[k]) for k in keys if k in best_row.index}

    result = BayeOptResult(
        optimizer=optimizer,
        scoreSummary=score_summary,
        best_params=best_params,
        best_score=best_score,
        config={
            **dict(cfg),
            "rodada.initPoints": init_points,
            "rodada.iters.n": iters_n,
            "rodada.acq": acq,
            "rodada.kernel": kernel_type,
            "rodada.bnxFile": bnx_file,
            "rodada.bnxLabel": bnx_label,
        },
    )

    valores_sufixo = [
        metodo_score,
        str(init_points),
        str(iters_n),
        acq,
        kernel_type,
        bnx_label,
    ]
    salvar_resultados_bo(result, output_dir, valores_sufixo)

    tempo_decorrido = calcular_tempo_dec(start_time)
    append_log(
        logfile,
        "*******************************************************\n"
        f"** Fim do processo de otimização com {valores_sufixo[0]}.\n"
        f"** Simulação PIniciais: {valores_sufixo[1]} Iterações: {valores_sufixo[2]}.\n"
        f"** Aquisição: {acq} | Kernel: {kernel_type} | BNX: {bnx_file}\n"
        f"** Tempo decorrido: {tempo_decorrido}\n\n"
    )

    return result

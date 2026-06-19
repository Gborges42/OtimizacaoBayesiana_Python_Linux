from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Union

from .parser import config_treatment, normalize_path_str


def _as_list_str(x: Any) -> List[str]:
    if x is None:
        return []
    if isinstance(x, list):
        return [str(v).strip() for v in x if str(v).strip() != ""]
    s = str(x).strip()
    return [s] if s else []


@dataclass(frozen=True)
class CalibConfig:
    # Paths principais
    dirExperiment: str
    dssatFile: str
    cultivarFile: str
    outputDir: str

    # Modelo/identificadores
    cultivar: str
    model: str

    # Calibração
    coefficients: List[str]
    limites: List[str]
    calibration: List[str]

    # Otimização
    initPoints: int
    iters_n: int
    iters_k: int
    acq: str
    metodo_score: str
    mape_iqr_lambda: float

    # Paralelo / semente
    paralelo: bool
    cores: int
    seed: int

    @staticmethod
    def from_dict(cfg: Mapping[str, Any]) -> "CalibConfig":
        # Required
        missing = [k for k in ("dirExperiment", "dssatFile", "cultivarFile", "outputDir", "cultivar", "model") if k not in cfg]
        if missing:
            raise ValueError(f"Campos obrigatórios ausentes no config: {missing}")

        dirExperiment = normalize_path_str(cfg["dirExperiment"])
        dssatFile = normalize_path_str(cfg["dssatFile"])
        cultivarFile = normalize_path_str(cfg["cultivarFile"])
        outputDir = normalize_path_str(cfg["outputDir"])

        cultivar = str(cfg["cultivar"])
        model = str(cfg["model"])

        coefficients = _as_list_str(cfg.get("coefficients"))
        limites = _as_list_str(cfg.get("limites"))
        calibration = _as_list_str(cfg.get("calibration"))

        initPoints = int(cfg.get("initPoints", 10))
        iters_n = int(cfg.get("iters.n", cfg.get("iters_n", 5)))
        iters_k = int(cfg.get("iters.k", cfg.get("iters_k", 1)))

        acq = str(cfg.get("acq", "ucb")).lower()
        metodo_score = str(cfg.get("metodo_score", "rmse")).lower()
        mape_iqr_lambda = float(cfg.get("mape_iqr_lambda", 0.5))

        paralelo = bool(cfg.get("paralelo", False))
        cores = max(int(cfg.get("cores", 1)), 1)
        seed = int(cfg.get("seed", 0))

        # validações úteis
        if metodo_score not in ("rmse", "mape"):
            raise ValueError(f"metodo_score inválido: {metodo_score}. Use 'rmse' ou 'mape'.")
        if acq not in ("ucb", "ei", "poi"):
            raise ValueError(f"acq inválido: {acq}. Use 'ucb', 'ei' ou 'poi'.")
        if initPoints < 1:
            raise ValueError("initPoints deve ser >= 1.")
        if iters_n < 1:
            raise ValueError("iters.n deve ser >= 1.")
        if iters_k < 1:
            raise ValueError("iters.k deve ser >= 1.")

        return CalibConfig(
            dirExperiment=dirExperiment,
            dssatFile=dssatFile,
            cultivarFile=cultivarFile,
            outputDir=outputDir,
            cultivar=cultivar,
            model=model,
            coefficients=coefficients,
            limites=limites,
            calibration=calibration,
            initPoints=initPoints,
            iters_n=iters_n,
            iters_k=iters_k,
            acq=acq,
            metodo_score=metodo_score,
            mape_iqr_lambda=mape_iqr_lambda,
            paralelo=paralelo,
            cores=cores,
            seed=seed,
        )

    @staticmethod
    def from_file(arq_config: Union[str, Path]) -> "CalibConfig":
        cfg = config_treatment(arq_config)
        return CalibConfig.from_dict(cfg)

    def validate_paths_exist(self) -> None:
        """
        Checagens boas pra rodar no Linux antes de começar BO.
        """
        if not Path(self.dirExperiment).exists():
            raise FileNotFoundError(f"dirExperiment não encontrado: {self.dirExperiment}")
        if not Path(self.dssatFile).exists():
            raise FileNotFoundError(f"dssatFile (executável) não encontrado: {self.dssatFile}")
        if not Path(self.cultivarFile).exists():
            raise FileNotFoundError(f"cultivarFile não encontrado: {self.cultivarFile}")
        Path(self.outputDir).mkdir(parents=True, exist_ok=True)

from __future__ import annotations

from pathlib import Path

from src.pipeline.run_best_configurations import run_best_configurations_from_summary


def main() -> None:
    run_best_configurations_from_summary(
        arq_config=Path("configs/StartValues_bean.config"),
        arq_resumo=Path("resumo_melhores_configuracoes.csv"),
        arq_saida=Path("evaluate_melhores_configuracoes.csv"),
    )


if __name__ == "__main__":
    main()
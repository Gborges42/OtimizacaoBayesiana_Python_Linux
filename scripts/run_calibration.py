from __future__ import annotations

from pathlib import Path
from src.pipeline.runner import run_grid_calibrations


def main():
    arq_config = Path("configs/StartValues_bean.config")
    run_grid_calibrations(arq_config)


if __name__ == "__main__":
    main()
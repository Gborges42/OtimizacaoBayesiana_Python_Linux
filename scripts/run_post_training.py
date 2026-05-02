from pathlib import Path

from src.config.parser import config_treatment
from src.pipeline.runner import build_input_list
from src.analysis.post_training import run_post_training_pipeline_for_bnx
from src.scoring.objective import simulation_function


def main():
    arq_config = Path("configs/StartValues_bean.config")

    cfg = config_treatment(arq_config)
    input_list = build_input_list(cfg)

    # Reaproveita lógica do runner
    bnx_files = cfg.get("bnxFiles")
    bnx_tests = cfg.get("bnxTest")

    if isinstance(bnx_files, str):
        bnx_files = [bnx_files]
    if isinstance(bnx_tests, str):
        bnx_tests = [bnx_tests]

    for bnx_file, bnx_test in zip(bnx_files, bnx_tests):
        bnx_label = Path(bnx_file).stem
        output_bnx_dir = Path("output") / bnx_label

        post_input = dict(input_list)
        post_input["bnxFile"] = bnx_file
        post_input["bnxTest"] = bnx_test
        post_input["bnxLabel"] = bnx_label

        print(f"Rodando pós-treinamento para {bnx_label}...")

        run_post_training_pipeline_for_bnx(
            output_bnx_dir=output_bnx_dir,
            input_list=post_input,
            simulation_function=simulation_function,
            bnx_train=bnx_file,
            bnx_test=bnx_test,
        )


if __name__ == "__main__":
    main()
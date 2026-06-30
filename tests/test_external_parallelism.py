from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

from src.pipeline import runner


class _InlineParallel:
    """Executa as tarefas joblib no processo do teste, preservando a ordem."""

    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, tasks):
        return [function(*args, **kwargs) for function, args, kwargs in tasks]


class ExternalParallelismTests(TestCase):
    def test_parallel_jobs_finish_before_post_training(self):
        events: list[str] = []

        with TemporaryDirectory() as tmp:
            cfg = {
                "initPoints": [10, 20],
                "iters.n": [1, 2],
                "acq": ["ucb", "ei"],
                "kernel": ["gauss", "matern3_2"],
                "bnxFiles": ["train.BNX"],
                "bnxTest": ["test.BNX"],
                "paralelo": True,
                "cores": 2,
                "outputDir": tmp,
            }

            def run_job(**job):
                events.append(f"job:{Path(job['logfile']).name}")

            def post_training(**kwargs):
                events.append("post")

            with (
                patch.object(runner, "config_treatment", return_value=cfg),
                patch.object(runner, "build_input_list", return_value={}),
                patch.object(runner, "_run_calibration_job", side_effect=run_job),
                patch.object(runner, "run_post_training_pipeline_for_bnx", side_effect=post_training),
                patch.object(runner, "Parallel", _InlineParallel),
                patch.object(runner, "parallel_config", side_effect=lambda **kwargs: nullcontext()),
            ):
                runner.run_grid_calibrations("unused.config")

            self.assertEqual(len(events), 3)
            self.assertTrue(events[0].startswith("job:001_ip-10_it-1"))
            self.assertTrue(events[1].startswith("job:002_ip-20_it-2"))
            self.assertEqual(events[2], "post")

            logs = sorted((Path(tmp) / "logs" / "train").glob("*.log"))
            self.assertEqual(len(logs), 2)
            self.assertNotEqual(logs[0].name, logs[1].name)

    def test_worker_disables_nested_objective_parallelism(self):
        with patch.object(runner, "run_simulation_baye") as run_baye:
            runner._run_calibration_job(
                combinacao={"initPoints": 1},
                cfg={"paralelo": True, "cores": 8},
                input_list={},
                logfile="worker.log",
            )

        kwargs = run_baye.call_args.kwargs
        self.assertIs(kwargs["objective_parallel"], False)
        self.assertEqual(kwargs["objective_cores"], 1)
        self.assertEqual(kwargs["cfg"]["cores"], 8)

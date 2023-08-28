import rootutils

root = rootutils.setup_root(__file__, dotenv=True, pythonpath=True, cwd=True)

import os
import subprocess
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from tests.utils import MLTypeEqualityCheckableMixin


class TestESTForGenerativeSequenceModelingLM(MLTypeEqualityCheckableMixin, unittest.TestCase):
    def setUp(self):
        os.environ["WANDB_MODE"] = "offline"
        os.environ["HYDRA_FULL_ERROR"] = "1"

        self.dir_objs = {}
        self.paths = {}
        for n in ("dataset", "pretraining/CI", "pretraining/NA", "from_scratch_finetuning", "RF"):
            self.dir_objs[n] = TemporaryDirectory()
            self.paths[n] = Path(self.dir_objs[n].name)

    def tearDown(self):
        for o in self.dir_objs.values():
            o.cleanup()

    def _test_command(self, command_parts: list[str], case_name: str):
        with self.subTest(case_name):
            command_out = subprocess.run(" ".join(command_parts), shell=True, capture_output=True)
            stderr = command_out.stderr.decode()
            stdout = command_out.stdout.decode()
            self.assertEqual(
                command_out.returncode, 0, f"Command errored!\nstderr:\n{stderr}\nstdout:\n{stdout}"
            )

    def build_dataset(self):
        command_parts = [
            "./scripts/build_dataset.py",
            f"--config-path='{(root / 'sample_data').resolve()}'",
            "--config-name=dataset",
            '"hydra.searchpath=[./configs]"',
            f"save_dir={self.paths['dataset']}",
        ]
        self._test_command(command_parts, "Build Dataset")

    def run_pretraining(self):
        cases = [
            {
                "msg": "Should be able to pre-train a conditionally independent model.",
                "config_name": "pretrain_CI",
                "save_dir": self.paths["pretraining/CI"],
            },
            {
                "msg": "Should be able to pre-train a nested attention model.",
                "config_name": "pretrain_NA",
                "save_dir": self.paths["pretraining/NA"],
            },
        ]

        for i, case in enumerate(cases):
            case_name = f"Pre-training case {i}: {case['msg']}"
            command_parts = [
                "./scripts/pretrain.py",
                f"--config-path='{(root / 'sample_data').resolve()}'",
                f"--config-name={case['config_name']}",
                '"hydra.searchpath=[./configs]"',
                f"data_config.save_dir={self.paths['dataset']}",
                f"experiment_dir={case['save_dir']}",
                f"save_dir={case['save_dir'] / 'model'}",
            ]

            self._test_command(command_parts, case_name)

    def run_finetuning(self):
        """Tests that fine-tuning can be run on a pre-trained model."""

        for task in (
            "multi_class_classification",
            "single_label_binary_classification",
            # "univariate_regression" Not currently supported.
        ):
            for finetune_from in ("pretraining/NA", "pretraining/CI"):
                command_parts = [
                    "./scripts/finetune.py",
                    f"data_config.save_dir={self.paths['dataset']}",
                    f"task_df_name={task}",
                    f"load_from_model_dir={self.paths[finetune_from] / 'model'}",
                    "optimization_config.max_epochs=2",
                ]
                self._test_command(command_parts, f"Fine-tuning from {finetune_from}: {task}")

    def run_from_scratch_training(self):
        """Tests that from-scratch supervised training can be run."""

        for task in (
            "multi_class_classification",
            "single_label_binary_classification",
            # "univariate_regression" Not currently supported.
        ):
            command_parts = [
                "./scripts/finetune.py",
                f"--config-path='{(root / 'sample_data').resolve()}'",
                "--config-name=from_scratch",
                '"hydra.searchpath=[./configs]"',
                f"data_config.save_dir={self.paths['dataset']}",
                f"experiment_dir={self.paths['from_scratch_finetuning']}",
                f"save_dir={self.paths['from_scratch_finetuning'] / task / 'model'}",
                f"task_df_name={task}",
            ]
            self._test_command(command_parts, f"From-scratch NN Training: {task}")

    def build_FT_task_df(self):
        command_parts = [
            "python",
            f"{(root/'sample_data'/'build_sample_task_DF.py').resolve()}",
            f"+dataset_dir={self.paths['dataset'].resolve()}",
        ]
        command_out = subprocess.run(" ".join(command_parts), shell=True, capture_output=True)

        if command_out.returncode != 0:
            raise ValueError(f"Building FT task dataframe failed: {command_out.stderr.decode()}")

    def test_e2e(self):
        # Data
        self.build_dataset()
        self.build_FT_task_df()

        # From-scratch training
        # self.run_from_scratch_training()

        # Pre-training
        self.run_pretraining()

        # Fine-tuning
        self.run_finetuning()


if __name__ == "__main__":
    unittest.main()

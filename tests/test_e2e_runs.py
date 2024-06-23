import rootutils

root = rootutils.setup_root(__file__, dotenv=True, pythonpath=True, cwd=True)

import json
import os
import subprocess
import unittest
from itertools import product
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import polars as pl

from tests.utils import MLTypeEqualityCheckableMixin


def dict_product(inp):
    return (dict(zip(inp.keys(), values)) for values in product(*inp.values()))


class TestESTForGenerativeSequenceModelingLM(MLTypeEqualityCheckableMixin, unittest.TestCase):
    TASKS = [
        "multi_class_classification",
        "single_label_binary_classification",
        # "univariate_regression" Not currently supported.
    ]

    def setUp(self):
        os.environ["WANDB_MODE"] = "offline"
        os.environ["HYDRA_FULL_ERROR"] = "1"

        self.dir_objs = {}
        self.paths = {}
        for n in (
            "dataset",
            "dataset_from_parquet",
            "esds",
            "pretraining/CI",
            "pretraining/NA",
            "from_scratch_finetuning",
            "sklearn",
        ):
            self.dir_objs[n] = TemporaryDirectory()
            self.paths[n] = Path(self.dir_objs[n].name)

    def tearDown(self):
        for o in self.dir_objs.values():
            o.cleanup()

    def _test_dataset_output(self, raw_data_root: Path, dataset_save_dir: Path):
        DL_save_dir = dataset_save_dir / "DL_reps"

        train_files = list((DL_save_dir / "train").glob("*.parquet"))
        tuning_files = list((DL_save_dir / "tuning").glob("*.parquet"))
        held_out_files = list((DL_save_dir / "held_out").glob("*.parquet"))

        assert len(set(train_files) & set(tuning_files)) == 0
        assert len(set(train_files) & set(held_out_files)) == 0
        assert len(set(tuning_files) & set(held_out_files)) == 0

        self.assertTrue(len(train_files) > 0)
        self.assertTrue(len(tuning_files) > 0)
        self.assertTrue(len(held_out_files) > 0)

        train_DL_reps = pl.concat([pl.read_parquet(f, use_pyarrow=False) for f in train_files])
        tuning_DL_reps = pl.concat([pl.read_parquet(f, use_pyarrow=False) for f in tuning_files])
        held_out_DL_reps = pl.concat([pl.read_parquet(f, use_pyarrow=False) for f in held_out_files])

        DL_shards = json.loads((dataset_save_dir / "DL_shards.json").read_text())

        ESD_subjects = pl.read_parquet(dataset_save_dir / "subjects_df.parquet", use_pyarrow=False)

        # Check that the DL shards are correctly partitioned.
        all_subjects = set(ESD_subjects["subject_id"].unique().to_list())

        self.assertEqual(len(all_subjects), len(ESD_subjects))

        all_subj_in_DL_shards = set().union(*DL_shards.values())

        all_subj_in_DL_shards = set(
            pl.Series(list(all_subj_in_DL_shards)).cast(ESD_subjects["subject_id"].dtype).to_list()
        )

        self.assertEqual(all_subjects, all_subj_in_DL_shards)

        all_train_DL_shard_subj = set().union(*(v for k, v in DL_shards.items() if k.startswith("train")))
        all_tuning_DL_shard_subj = set().union(*(v for k, v in DL_shards.items() if k.startswith("tuning")))
        all_held_out_DL_shard_subj = set().union(
            *(v for k, v in DL_shards.items() if k.startswith("held_out"))
        )

        self.assertEqual(len(all_train_DL_shard_subj & all_tuning_DL_shard_subj), 0)
        self.assertEqual(len(all_train_DL_shard_subj & all_held_out_DL_shard_subj), 0)
        self.assertEqual(len(all_tuning_DL_shard_subj & all_held_out_DL_shard_subj), 0)

        train_DL_subjects = set(train_DL_reps["subject_id"].to_list())
        tuning_DL_subjects = set(tuning_DL_reps["subject_id"].to_list())
        held_out_DL_subjects = set(held_out_DL_reps["subject_id"].to_list())

        self.assertEqual(all_train_DL_shard_subj, {str(x) for x in train_DL_subjects})
        self.assertEqual(all_tuning_DL_shard_subj, {str(x) for x in tuning_DL_subjects})
        self.assertEqual(all_held_out_DL_shard_subj, {str(x) for x in held_out_DL_subjects})

        self.assertTrue(len(train_DL_subjects) > len(tuning_DL_subjects))
        self.assertTrue(len(train_DL_subjects) > len(held_out_DL_subjects))

        all_DL_subjects = train_DL_subjects | tuning_DL_subjects | held_out_DL_subjects

        self.assertEqual(all_DL_subjects, all_subjects)

    def _test_command(self, command_parts: list[str], case_name: str, use_subtest: bool = True):
        if use_subtest:
            with self.subTest(case_name):
                command_out = subprocess.run(" ".join(command_parts), shell=True, capture_output=True)
                stderr = command_out.stderr.decode()
                stdout = command_out.stdout.decode()
                self.assertEqual(
                    command_out.returncode, 0, f"Command errored!\nstderr:\n{stderr}\nstdout:\n{stdout}"
                )
        else:
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
        self._test_command(command_parts, "Build Dataset", use_subtest=False)
        self._test_dataset_output((root / "sample_data" / "raw"), self.paths["dataset"])

        command_parts = [
            "./scripts/build_dataset.py",
            f"--config-path='{(root / 'sample_data').resolve()}'",
            "--config-name=dataset_parquet",
            '"hydra.searchpath=[./configs]"',
            f"save_dir={self.paths['dataset_from_parquet']}",
        ]
        self._test_command(command_parts, "Build Dataset from Parquet", use_subtest=False)
        self._test_dataset_output((root / "sample_data" / "raw"), self.paths["dataset_from_parquet"])

    def build_ESDS_dataset(self):
        command_parts = [
            "./scripts/convert_to_ESDS.py",
            f"dataset_dir={self.paths['dataset']}",
            f"ESDS_save_dir={self.paths['esds']}",
            "ESDS_chunk_size=25",
        ]
        self._test_command(command_parts, "Build ESDS Dataset", use_subtest=True)

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

            self._test_command(command_parts, case_name, use_subtest=False)

    def run_finetuning(self):
        """Tests that fine-tuning can be run on a pre-trained model."""

        for task in self.TASKS:
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

        for task in self.TASKS:
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

    def run_get_embeddings(self):
        task = "multi_class_classification"  # Get embeddings is not sensitive to task.
        for get_embeddings_from in ("pretraining/NA", "pretraining/CI"):
            command_parts = [
                "./scripts/get_embeddings.py",
                f"data_config.save_dir={self.paths['dataset']}",
                f"task_df_name={task}",
                f"load_from_model_dir={self.paths[get_embeddings_from] / 'model'}",
                "optimization_config.max_epochs=2",
            ]
            self._test_command(command_parts, f"Get embeddings from {get_embeddings_from}: {task}")

    def run_sklearn_baseline(self):
        def make_command(
            groups_and_options: dict[str, tuple[str, dict[str, Any]]],
            task: str,
        ) -> list[str]:
            cmd = [
                "./scripts/sklearn_baseline.py",
                f"dataset_dir={self.paths['dataset'].resolve()}",
                f"experiment_dir={self.paths['sklearn'].resolve()}",
                f"task_df_name={task}",
                "finetuning_task_label=label",
                "wandb_logger_kwargs.name=null",
                "wandb_logger_kwargs.save_code=False",
            ]

            for n, gp_options in groups_and_options.items():
                match gp_options:
                    case [str(), dict()]:
                        gp, options = gp_options
                    case str():
                        gp = gp_options
                        options = {}
                    case _:
                        raise ValueError(f"{gp_options} of type {type(gp_options)} malformed!")

                cmd.append(f"{n}={gp}")
                for k, v in options.items():
                    match v:
                        case list():
                            v_str = f"[{','.join(v)}]"
                        case _:
                            v_str = str(v)
                    cmd.append(f"{n}.{k}={v_str}")

            return cmd

        cfg_options = {
            "task": self.TASKS,
            "scaling": ("min_max_scaler", "standard_scaler"),
            "imputation": ("simple_imputer", "knn_imputer"),
            "dim_reduce": (
                "pca",
                "select_k_best",
            ),  # 'nmf' doesn't universally work, as input must be non-neg
            "feature_selector": (
                (
                    "esd_flat_feature_loader",
                    {
                        "window_sizes": ["1h", "1d", "FULL", "-1h", "-FULL"],
                        "feature_inclusion_frequency": 1e-3,
                    },
                ),
            ),
            "model": (("random_forest_classifier", {"n_estimators": 2}),),
        }

        for task in cfg_options.pop("task"):
            for cfg in dict_product(cfg_options):
                cmd = make_command(cfg, task)
                self._test_command(cmd, f"Sklearn for {' '.join(cmd)}")

    def run_zeroshot(self):
        classification_labeler_path = root / "sample_data" / "sample_classification_labeler.py"
        for task in self.TASKS:
            task_labeler_path = self.paths["dataset"] / "task_dfs" / f"{task}_labeler.py"
            cp_parts = ["cp", str(classification_labeler_path.resolve()), str(task_labeler_path.resolve())]

            cp_out = subprocess.run(" ".join(cp_parts), shell=True, capture_output=True)

            if cp_out.returncode != 0:
                raise ValueError(f"Copying {task} labeler failed: {cp_out.stderr.decode()}")

        for task in self.TASKS:
            for zeroshot_from in ("pretraining/NA", "pretraining/CI"):
                zeroshot_command_parts = [
                    "./scripts/zeroshot.py",
                    f"data_config.save_dir={self.paths['dataset']}",
                    "data_config.do_include_subject_id=True",
                    "data_config.do_include_subsequence_indices=True",
                    "data_config.do_include_start_time_min=True",
                    "data_config.max_seq_len=32",
                    "config.task_specific_params.num_samples=3",
                    "data_config.seq_padding_side=left",
                    f"task_df_name={task}",
                    f"load_from_model_dir={self.paths[zeroshot_from] / 'model'}",
                ]
                self._test_command(zeroshot_command_parts, f"Zero-shot for {task} from {zeroshot_from}")

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
        self.build_ESDS_dataset()
        self.build_FT_task_df()

        # Sklearn baselines
        self.run_sklearn_baseline()

        # From-scratch training
        self.run_from_scratch_training()

        # Pre-training
        self.run_pretraining()

        # Fine-tuning
        self.run_finetuning()

        # Get embeddings
        self.run_get_embeddings()

        # Zero-shot
        self.run_zeroshot()


if __name__ == "__main__":
    unittest.main()

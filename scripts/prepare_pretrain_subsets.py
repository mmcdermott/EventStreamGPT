#!/usr/bin/env python
"""Prepares subset directories, config files, and commands for running experiments.

Includes:
    * Pre-training subset experiments (for a specified architecture).
    * Fine-tuning over few-shot subsets on specifiable fine-tuning tasks.
    * Performing zero-shot evaluation of a model over a set of fine-tuning tasks.
    * Getting model embeedings.
"""

try:
    # This color-codes and prettifies error messages if the script fails.
    import stackprinter

    stackprinter.set_excepthook(style="darkbg2")
except ImportError:
    pass  # no need to fail because of missing dev dependency

import copy
from collections import defaultdict
from pathlib import Path

import hydra
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from EventStream.data.config import SeqPaddingSide, SubsequenceSamplingStrategy
from EventStream.logger import hydra_loguru_init


@hydra.main(version_base=None, config_path="../configs", config_name="pretrain_subsets_base")
def main(cfg: DictConfig):
    hydra_loguru_init()
    cfg = hydra.utils.instantiate(cfg, _convert_="all")

    # Validation
    initial_model_path = Path(cfg["initial_model_path"])
    initial_config_path = initial_model_path / "pretrain_config.yaml"
    if not initial_config_path.is_file():
        raise FileNotFoundError(f"{initial_config_path} does not exist!")

    subset_sizes = cfg["subset_sizes"]
    if not isinstance(subset_sizes, list):
        raise TypeError(f"subset_sizes must be a list, got {subset_sizes}!")

    seeds = cfg["seeds"]
    match seeds:
        case int():
            seeds = [seeds for _ in subset_sizes]
        case list() if len(seeds) == len(subset_sizes):
            pass
        case dict() if all([subset in seeds for subset in subset_sizes]):
            seeds = [seeds[subset] for subset in subset_sizes]
        case _:
            raise TypeError(f"seeds must be an int or a list/dict matching {subset_sizes}, got {seeds}!")

    # Load initial config information
    initial_config = OmegaConf.load(initial_config_path)

    experiment_dir = cfg["experiment_dir"]
    if experiment_dir is None:
        experiment_dir = initial_config.experiment_dir
        logger.info(f"Setting experiment dir to {experiment_dir}!")

    experiment_dir = Path(experiment_dir)

    runs_dir = experiment_dir / cfg["experiment_name"]
    runs_dir.mkdir(parents=True, exist_ok=True)

    do_FT_commands = len(cfg["few_shot_commands"]["fine_tuning_task_names"]) > 0
    do_zero_shot_commands = len(cfg["zero_shot_commands"]["fine_tuning_task_names"]) > 0
    do_embeddings_commands = len(cfg["get_embeddings_commands"]["fine_tuning_task_names"]) > 0

    # Create subset experiment directories and run commands
    commands = defaultdict(list)
    for n_seeds, subset_size in zip(seeds, subset_sizes):
        subset_runs_dir = runs_dir / f"subset_{subset_size}"
        for seed in range(n_seeds):
            seed_runs_dir = subset_runs_dir / f"seed_{seed}"
            seed_runs_dir.mkdir(parents=True, exist_ok=True)

            if cfg["do_include_PT_commands"]:
                new_config = copy.deepcopy(initial_config)
                new_config["defaults"] = ["pretrain_config", "_self_"]
                new_config.experiment_dir = experiment_dir
                new_config.data_config.train_subset_size = subset_size
                new_config.data_config.train_subset_seed = seed
                new_config.save_dir = str(seed_runs_dir)
                new_config.wandb_logger_kwargs.name = f"pretrain_subset_{subset_size}_seed_{seed}"
                new_config.wandb_logger_kwargs.project = cfg["project"]
                new_config.wandb_experiment_config_kwargs = {
                    "save_dir": str(seed_runs_dir),
                    "subset_size": subset_size,
                    "subset_seed": seed,
                    "experiment_name": cfg["experiment_name"],
                }

                new_config_path = seed_runs_dir / "pretrain_config_source.yaml"
                OmegaConf.save(new_config, new_config_path)

                # make command
                command = (
                    'PYTHONPATH="$EVENT_STREAM_PATH:$PYTHONPATH" '
                    "python $EVENT_STREAM_PATH/scripts/pretrain.py "
                    f"--config-path={seed_runs_dir} "
                    f"--config-name=pretrain_config_source "
                    f"hydra.searchpath=[$EVENT_STREAM_PATH/configs]"  # Ensures hydra instantiates correctly
                )
                commands["PT"].append(command)

            if do_embeddings_commands:
                for FT_task in cfg["get_embeddings_commands"]["fine_tuning_task_names"]:
                    get_embeddings_config = dict(
                        defaults=["finetune_config", "_self_"],
                        load_from_model_dir=str(seed_runs_dir),
                        do_overwrite=False,
                        task_df_name=FT_task,
                        data_config={
                            "subsequence_sampling_strategy": str(SubsequenceSamplingStrategy.TO_END),
                            "train_subset_size": "FULL",
                            "train_subset_seed": None,
                        },
                        config={"task_specific_params": {"pooling_method": "last"}},
                        optimization_config=dict(**cfg["get_embeddings_commands"]["optimization_config"]),
                    )

                    get_embeddings_config_path = (
                        seed_runs_dir / "embeddings" / FT_task / "get_embeddings_config_source.yaml"
                    )
                    get_embeddings_config_path.parent.mkdir(exist_ok=True, parents=True)

                    OmegaConf.save(get_embeddings_config, get_embeddings_config_path)

                    command = (
                        'PYTHONPATH="$EVENT_STREAM_PATH:$PYTHONPATH" '
                        "python $EVENT_STREAM_PATH/scripts/get_embeddings.py "
                        f"--config-path={get_embeddings_config_path.parent} "
                        f"--config-name=get_embeddings_config_source "
                        f"hydra.searchpath=[$EVENT_STREAM_PATH/configs]"
                    )
                    commands["get_embeddings"].append(command)

            if do_zero_shot_commands:
                zero_shot_dir = seed_runs_dir / "zero_shot"
                for FT_task in cfg["zero_shot_commands"]["fine_tuning_task_names"]:
                    zero_shot_task_dir = zero_shot_dir / FT_task
                    zero_shot_task_dir.mkdir(parents=True, exist_ok=True)

                    zero_shot_config = dict(
                        defaults=["finetune_config", "_self_"],
                        load_from_model_dir=str(seed_runs_dir),
                        save_dir=zero_shot_task_dir,
                        do_overwrite=False,
                        task_df_name=FT_task,
                        data_config={
                            "subsequence_sampling_strategy": str(SubsequenceSamplingStrategy.TO_END),
                            "seq_padding_side": str(SeqPaddingSide.LEFT),
                            "max_seq_len": cfg["zero_shot_commands"]["input_seq_len"],
                            "do_include_start_time_min": True,
                        },
                        config={
                            "task_specific_params": {
                                "num_samples": cfg["zero_shot_commands"]["num_samples"],
                            }
                        },
                        optimization_config=dict(**cfg["zero_shot_commands"]["optimization_config"]),
                        wandb_logger_kwargs={
                            "name": f"zero_shot_{FT_task}_PT_{subset_size}_seed_{seed}",
                            "project": cfg["project"],
                        },
                        wandb_experiment_config_kwargs={
                            "FT_task": FT_task,
                            "save_dir": str(zero_shot_task_dir),
                            "FT_subset_size": 0,
                            "PT_subset_size": subset_size,
                            "subset_seed": seed,
                            "experiment_name": f"{cfg['experiment_name']}/zero_shot",
                            "PT_model_dir": str(seed_runs_dir),
                        },
                    )

                    OmegaConf.save(
                        zero_shot_config,
                        zero_shot_task_dir / "zero_shot_config_source.yaml",
                    )

                    command = (
                        'PYTHONPATH="$EVENT_STREAM_PATH:$PYTHONPATH" '
                        "python $EVENT_STREAM_PATH/scripts/zeroshot.py "
                        f"--config-path={zero_shot_task_dir} "
                        f"--config-name=zero_shot_config_source "
                        f"hydra.searchpath=[$EVENT_STREAM_PATH/configs]"
                    )

                    commands["zero_shot"].append(command)

            if do_FT_commands:
                FT_dir = seed_runs_dir / "fine_tuning"
                for FT_task in cfg["few_shot_commands"]["fine_tuning_task_names"]:
                    FT_task_dir = FT_dir / FT_task
                    for FT_subset_size in cfg["few_shot_commands"]["fine_tuning_subset_sizes"]:
                        FT_subset_dir = FT_task_dir / f"subset_{FT_subset_size}"
                        FT_subset_dir.mkdir(parents=True, exist_ok=True)

                        FT_config = dict(
                            defaults=["finetune_config", "_self_"],
                            load_from_model_dir=str(seed_runs_dir),
                            save_dir=FT_subset_dir,
                            do_overwrite=False,
                            task_df_name=FT_task,
                            data_config={
                                "subsequence_sampling_strategy": str(SubsequenceSamplingStrategy.TO_END),
                                "train_subset_size": FT_subset_size,
                                "train_subset_seed": seed,
                            },
                            config={"task_specific_params": {"pooling_method": "last"}},
                            optimization_config=dict(**cfg["few_shot_commands"]["optimization_config"]),
                            wandb_logger_kwargs={
                                "name": (
                                    f"finetune_{FT_task}_{FT_subset_size}_shot_PT_{subset_size}_seed_{seed}"
                                ),
                                "project": cfg["project"],
                            },
                            wandb_experiment_config_kwargs={
                                "FT_task": FT_task,
                                "save_dir": str(FT_subset_dir),
                                "FT_subset_size": FT_subset_size,
                                "PT_subset_size": subset_size,
                                "subset_seed": seed,
                                "experiment_name": f"{cfg['experiment_name']}/fine_tuning",
                                "PT_model_dir": str(seed_runs_dir),
                            },
                        )

                        OmegaConf.save(
                            FT_config,
                            FT_subset_dir / "finetune_config_source.yaml",
                        )

                        command = (
                            'PYTHONPATH="$EVENT_STREAM_PATH:$PYTHONPATH" '
                            "python $EVENT_STREAM_PATH/scripts/finetune.py "
                            f"--config-path={FT_subset_dir} "
                            f"--config-name=finetune_config_source "
                            f"hydra.searchpath=[$EVENT_STREAM_PATH/configs]"
                        )

                        commands["FT"].append(command)

    # Save commands to file
    for key, value in commands.items():
        commands_path = runs_dir / f"{key}_commands.txt"
        with open(commands_path, "w") as f:
            f.write("\n".join(value))
        logger.info(f"{key} Commands written to {commands_path}!")


if __name__ == "__main__":
    main()

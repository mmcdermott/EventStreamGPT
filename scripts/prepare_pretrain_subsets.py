#!/usr/bin/env python

try:
    import stackprinter

    stackprinter.set_excepthook(style="darkbg2")
except ImportError:
    pass  # no need to fail because of missing dev dependency

import copy
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base=None, config_path="../configs", config_name="pretrain_subsets_base")
def main(cfg: DictConfig):
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
            raise TypeError(
                f"seeds must be an int or a list/dict matching {subset_sizes}, got {seeds}!"
            )

    # Load initial config information
    initial_config = OmegaConf.load(initial_config_path)

    experiment_dir = cfg["experiment_dir"]
    if experiment_dir is None:
        experiment_dir = initial_config.experiment_dir
        print(f"Setting experiment dir to {experiment_dir}!")

    experiment_dir = Path(experiment_dir)

    runs_dir = experiment_dir / cfg["experiment_name"]
    runs_dir.mkdir(parents=True, exist_ok=True)

    # Create subset experiment directories and run commands
    commands = []
    for n_seeds, subset_size in zip(seeds, subset_sizes):
        subset_runs_dir = runs_dir / f"subset_{subset_size}"
        for seed in range(n_seeds):
            seed_runs_dir = subset_runs_dir / f"seed_{seed}"
            seed_runs_dir.mkdir(parents=True, exist_ok=True)

            new_config = copy.deepcopy(initial_config)
            new_config['defaults'] = ['pretrain_config', '_self_']
            new_config.experiment_dir = experiment_dir
            new_config.data_config.train_subset_size = subset_size
            new_config.data_config.train_subset_seed = seed
            new_config.save_dir = str(seed_runs_dir)
            new_config.wandb_logger_kwargs.name = f"pretrain_subset_{subset_size}_seed_{seed}"
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
                'PYTHONPATH="$EVENT_STREAM_PATH:$PYTHONPATH" python $EVENT_STREAM_PATH/scripts/pretrain.py '
                f"--config-path={seed_runs_dir} "
                f"--config-name=pretrain_config_source "
                f"hydra.searchpath=[$EVENT_STREAM_PATH/configs]" # Ensures hydra instantiates correctly
            )
            commands.append(command)

    # Save commands to file
    commands_path = runs_dir / "commands.txt"
    with open(commands_path, "w") as f:
        f.write("\n".join(commands))
    print(f"Commands written to {commands_path}!")
    print("Commands:")
    print("\n".join(commands))


if __name__ == "__main__":
    main()

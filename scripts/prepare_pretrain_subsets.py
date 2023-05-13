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

    experiment_out_dir = Path(cfg["experiment_out_dir"])
    experiment_out_dir.mkdir(parents=True, exist_ok=True)

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

    # Create subset experiment directories and run commands
    commands = []
    for n_seeds, subset_size in zip(seeds, subset_sizes):
        subset_experiment_out_dir = experiment_out_dir / f"subset_{subset_size}"
        for seed in range(n_seeds):
            seed_experiment_out_dir = subset_experiment_out_dir / f"seed_{seed}"
            seed_experiment_out_dir.mkdir(parents=True, exist_ok=True)

            new_config = copy.deepcopy(initial_config)
            new_config.data_config.train_subset_size = subset_size
            new_config.data_config.train_subset_seed = seed

            new_config_path = seed_experiment_out_dir / "pretrain_config.yaml"
            OmegaConf.save(new_config, new_config_path)

            # make command
            command = (
                'PYTHONPATH="$EVENT_STREAM_PATH:$PYTHONPATH" python $EVENT_STREAM_PATH/scripts/pretrain.py '
                f"--config-path={seed_experiment_out_dir} "
                f"--config-name=pretrain_config "
            )
            commands.append(command)

    # Save commands to file
    commands_path = experiment_out_dir / "commands.txt"
    with open(commands_path, "w") as f:
        f.write("\n".join(commands))
    print(f"Commands written to {commands_path}!")
    print("Commands:")
    print("\n".join(commands))


if __name__ == "__main__":
    main()

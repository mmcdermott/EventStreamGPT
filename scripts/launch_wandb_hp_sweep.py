import hydra
import wandb
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="../configs", config_name="hyperparameter_sweep")
def main(cfg: DictConfig):
    cfg = hydra.utils.instantiate(cfg, _convert_="all")
    cfg["command"] = [
        "WANDB_START_METHOD=thread",
        "${env}",
        "${interpreter}",
        "${program}",
        "${args_no_hyphens}",
    ]

    sweep_id = wandb.sweep(sweep=cfg)
    print(f"Created sweep with ID: {sweep_id}")
    return sweep_id


if __name__ == "__main__":
    main()

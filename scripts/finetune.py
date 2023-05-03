import sys
sys.path.append('..')

import hydra

from EventStream.transformer.stream_classification_lightning import FinetuneConfig, train

@hydra.main(version_base=None, config_name="finetune_config")
def main(cfg: FinetuneConfig):
    if type(cfg) is not FinetuneConfig:
        cfg = hydra.utils.instantiate(cfg, _convert_='object')
    # TODO(mmd): This isn't the right return value for hyperparameter sweeps.
    return train(cfg)

if __name__ == "__main__":
    main()

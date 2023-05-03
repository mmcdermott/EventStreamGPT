try:
    # import pretty_traceback
    # pretty_traceback.install()

    import stackprinter

    stackprinter.set_excepthook(style="darkbg2")
except ImportError:
    pass  # no need to fail because of missing dev dependency

import hydra
import torch

torch.set_float32_matmul_precision("high")

from EventStream.transformer.generative_sequence_modelling_lightning import (
    PretrainConfig,
    train,
)


@hydra.main(version_base=None, config_name="pretrain_config")
def main(cfg: PretrainConfig):
    if type(cfg) is not PretrainConfig:
        cfg = hydra.utils.instantiate(cfg, _convert_="object")
    # TODO(mmd): This isn't the right return value for hyperparameter sweeps.
    return train(cfg)


if __name__ == "__main__":
    main()

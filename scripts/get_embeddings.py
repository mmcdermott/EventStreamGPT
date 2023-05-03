import hydra

from EventStream.transformer.get_embeddings_lightning import (
    GetEmbeddingsConfig,
    get_embeddings,
)


@hydra.main(version_base=None, config_name="get_embeddings_config")
def main(cfg: GetEmbeddingsConfig):
    if type(cfg) is not GetEmbeddingsConfig:
        cfg = hydra.utils.instantiate(cfg, _convert_="object")
    # TODO(mmd): This isn't the right return value for hyperparameter sweeps.
    return get_embeddings(cfg)


if __name__ == "__main__":
    main()

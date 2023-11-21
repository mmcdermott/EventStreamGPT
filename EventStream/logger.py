import os
import hydra
from loguru import logger as log

@hydra.main(version_base=None)
def hydra_loguru_init(_) -> None:
    hydra_path = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    log.add(os.path.join(hydra_path, "main.log"))

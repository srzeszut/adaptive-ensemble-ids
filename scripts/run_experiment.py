import hydra
from omegaconf import DictConfig, OmegaConf

from src.utils.silence import apply_silence

apply_silence()

from src.pipeline.runner import run_pipeline


@hydra.main(version_base=None, config_path="../configs", config_name="offline")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    run_pipeline(cfg)


if __name__ == "__main__":
    main()

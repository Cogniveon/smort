import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig


@hydra.main(config_path="configs", config_name="app", version_base="1.3")
def run_server(cfg: DictConfig):
    server = instantiate(cfg.server)
    server.run(debug=cfg.debug)


if __name__ == "__main__":
    run_server()

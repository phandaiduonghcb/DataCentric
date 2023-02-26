import logging
from pathlib import Path
import random
import numpy as np
import hydra
import pytorch_lightning as pl
import torch
import wandb
import random
import numpy as n
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import Callback, ModelCheckpoint, RichProgressBar, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy
from torch.distributed.algorithms.ddp_comm_hooks import default_hooks as default

# from src.dataset_s import LivenessDatamodule
from model_s import TIMMModel
from dataset_s import CustomDataModule

log = logging.getLogger(__name__)
wandb.login(key='ff6d8c7f99c93d809cf8a344fe9dd48506a40e3a')
def test(config):
    wandb_logger = WandbLogger(
        project="zalo_2022",
        log_model=False,
        settings=wandb.Settings(start_method="spawn"),
        name=Path.cwd().stem,
        dir=Path.cwd()
    )
    
    OmegaConf.set_struct(config, True)
    strategy = config.trainer.strategy
    print(f'Strategy: {strategy}')
    if strategy == "ddp" and config.trainer.accelerator == "gpu":
        if config.trainer.devices == -1:
            config.trainer.devices = torch.cuda.device_count()

        num_nodes = getattr(config.trainer, "num_nodes", 1)
        total_gpus = max(1, config.trainer.devices * num_nodes)
        config.dataset.batch_size = int(config.dataset.batch_size / total_gpus)
        config.dataset.num_workers = int(config.dataset.num_workers / total_gpus)
        strategy = DDPStrategy(
            find_unused_parameters=config.ddp_plugin.find_unused_params,
            gradient_as_bucket_view=True,
            ddp_comm_hook=default.fp16_compress_hook
            if config.ddp_plugin.fp16_hook
            else None,
            static_graph=config.ddp_plugin.static_graph,
        )
    run = wandb.init()
    artifact = run.use_artifact(config.test.wandb_saved_checkpoint, type='model')
    artifact_dir = artifact.download()
    model = TIMMModel.load_from_checkpoint(Path(artifact_dir) / "model.ckpt")
    model.eval()
    datamodule = CustomDataModule(config.dataset)
    datamodule.setup()
    trainer = pl.Trainer(
        logger=wandb_logger,
        **config.trainer,
    )

    wandb_logger.watch(model, log="parameters", log_graph=False)
#     predictions = trainer.predict(model, datamodule.val_dataloader())
    trainer.test(model, datamodule=datamodule)
    print(config.test.wandb_saved_checkpoint)
    print(Path(artifact_dir) / "model.ckpt")
    wandb.finish()

@hydra.main(config_path="configs", config_name="baseline_s", version_base='1.1')
def main(config: DictConfig):
    log.info("Zalo AI Challenge - Liveness Detection")
    log.info(f"Current working directory : {Path.cwd()}")
    test(config)

if __name__ == "__main__":
    main()
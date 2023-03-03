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
import glob
import os

# from src.dataset_s import LivenessDatamodule
from model_s import TIMMModel
from dataset_s import CustomDataModule

log = logging.getLogger(__name__)
wandb.login(key='ff6d8c7f99c93d809cf8a344fe9dd48506a40e3a')

def train(config):
    wandb_logger = WandbLogger(
        project="DataCentric",
        log_model=True,
        settings=wandb.Settings(start_method="spawn"),
        name=Path.cwd().stem,
        dir=Path.cwd()
    )
    
    # Create callbacks
    callbacks = []
    callbacks.append(ModelCheckpoint(**config.model_ckpt))
    callbacks.append(RichProgressBar(config.refresh_rate))
    callbacks.append(LearningRateMonitor(logging_interval='epoch'))

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

    model = TIMMModel(config.model)
    datamodule = CustomDataModule(config.dataset)
    trainer = pl.Trainer(
        logger=wandb_logger,
        callbacks=callbacks,
        detect_anomaly=True,
        # strategy=strategy,
        **config.trainer,
    )

    wandb_logger.watch(model, log="parameters", log_graph=False)
    trainer.fit(model, datamodule=datamodule)
    # best_ckpt_path = glob.glob(os.path.join(os.path.join(os.getcwd(), config.model_ckpt.dirpath), "checkpoint*"))[0]
    # best_ckpt_path = "/DataCentric/trained_models/raw_augmented.ckpt"
    # model = TIMMModel.load_from_checkpoint(best_ckpt_path)
    model.eval()
    trainer.test(model, datamodule=datamodule,ckpt_path='best')
    columns = ["true","pred", "image","path"]
    classes = np.array(datamodule.test_dataset.dataset.classes)
    for i in range(len(model.test_table)):
        model.test_table[i][0] = classes[model.test_table[i][0]]
        model.test_table[i][1] = classes[model.test_table[i][1]]
        
    model.logger.log_table(key='wrong_pred_images', columns=columns, data= model.test_table)
    predictions = trainer.predict(model, datamodule.val_dataloader(), ckpt_path='best')
    predictions = torch.cat(predictions)
    wandb.log({"Confusion matrix": wandb.plot.confusion_matrix(
        preds=predictions.cpu().numpy(), y_true = model.predicted_targets.cpu().numpy(),
        class_names= classes
    )})
    print(config.dataset.train_data_dir)
    print(config.model_ckpt.dirpath)
    wandb.finish()

def test(config):
    wandb_logger = WandbLogger(
        project="DataCentric",
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

    print(config.test.saved_checkpoint_path)
    model = TIMMModel.load_from_checkpoint(config.test.saved_checkpoint_path)
    # best_ckpt_path = glob.glob(os.path.join(os.path.join(os.getcwd(), config.model_ckpt.dirpath), "checkpoint*"))[0]
    # model = TIMMModel.load_from_checkpoint(best_ckpt_path)
    model.eval()
    datamodule = CustomDataModule(config.dataset)
    datamodule.setup()
    trainer = pl.Trainer(
        logger=wandb_logger,
        **config.trainer,
    )

    wandb_logger.watch(model, log="parameters", log_graph=False)
    trainer.test(model, datamodule=datamodule)
    columns = ["true","pred", "image","path"] 
    columns = ["true","classes", "probs","image"] #
    classes = np.array(datamodule.test_dataset.dataset.classes)
    for i in range(len(model.test_table)):
        model.test_table[i][0] = classes[model.test_table[i][0]]
        model.test_table[i][1] = np.array(classes)[model.test_table[i][1].cpu().numpy()]
        model.test_table[i] = model.test_table[i][:-1]
        
    model.logger.log_table(key='wrong_pred_images', columns=columns, data= model.test_table)
    predictions = trainer.predict(model, datamodule.val_dataloader())
    predictions = torch.cat(predictions)
    # wandb.log({"Confusion matrix": wandb.plot.confusion_matrix(
    #     preds=predictions.cpu().numpy(), y_true = model.predicted_targets.cpu().numpy(),
    #     class_names= classes
    # )})
    
    print(config.test.saved_checkpoint_path)
    wandb.finish()

@hydra.main(config_path="configs", config_name="baseline_s", version_base='1.1')
def main(config: DictConfig):
    log.info("Zalo AI Challenge - Liveness Detection")
    log.info(f"Current working directory : {Path.cwd()}")
    if config.state == "train":
        train(config)
    elif config.state == "test":
        test(config)

if __name__ == "__main__":
    main()
    
# Based on Yet Another Lightning Hydra Template (https://github.com/gorodnitskiy/yet-another-lightning-hydra-template)
from data.utils import load_normalize_from_file, create_omegaconf_from_json
from utils.logging_utils import log_hyperparameters
from utils.pylogger import get_pylogger
from utils.env_utils import instantiate_callbacks, instantiate_loggers
from lightning.pytorch import (
    Callback,
    LightningDataModule,
    LightningModule,
    Trainer,
    seed_everything,
)
from omegaconf import DictConfig, OmegaConf
import torch.multiprocessing
import hydra
from models.sup_cont import ContrastiveResNet50
from bolts.lr_scheduler import LinearWarmupCosineAnnealingLR
from typing import List, Optional, Tuple
import os


import datetime
import lightning

import numpy as np
np.float_ = np.float64


_HYDRA_PARAMS = {
    "version_base": "1.3",
    "config_path": "configs",  # os.path.join(os.getcwd(), "/configs"),
    "config_name": "train.yaml",
}
log = get_pylogger(__name__)


os.environ["PROJECT_ROOT"] = os.path.dirname(os.path.abspath(__file__))

OmegaConf.register_new_resolver("len", len)
OmegaConf.register_new_resolver(
    "load_normalize_from_file", load_normalize_from_file)

OmegaConf.register_new_resolver(
    "create_omegaconf_from_json", create_omegaconf_from_json)


def train(cfg: DictConfig) -> Tuple[dict, dict]:
    """Trains the model. Can additionally evaluate on a testset, using best
    weights obtained during training.

    Args:
        cfg (DictConfig): Configuration composed by Hydra.

    Returns:
        Tuple[dict, dict]: Dict with metrics and dict with all instantiated
        objects.
    """

    # set seed for random number generators in pytorch, numpy and python.random

    if cfg.get("seed"):
        log.info(f"Seed everything with <{cfg.seed}>")
        seed_everything(cfg.seed, workers=True)

    # transforms

    log.info(cfg.get("module"))

    train_transform = hydra.utils.instantiate(
        cfg.train_transform,
        _recursive_=True,
        _convert_="all"
    )
    val_transform = hydra.utils.instantiate(
        cfg.val_transform,
        _recursive_=True,
        _convert_="all"
    )

    # Init lightning datamodule
    log.info(f"Instantiating datamodule <{cfg.data.data_module._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(
        cfg.data.data_module, _recursive_=False,
    )

    datamodule = datamodule(
        train_transform=train_transform, val_transform=val_transform)

    if cfg.get("load_model_from_ckpt", None) is not None:
        log.info(f"Loading model from checkpoint <{cfg.load_model_from_ckpt}>")
        model: LightningModule = ContrastiveResNet50.load_from_checkpoint(
            cfg.load_model_from_ckpt
        )
        print(cfg.module.weighted_sampling_config.start_epoch)
        model.weighted_sampling_config.start_epoch = cfg.module.weighted_sampling_config.start_epoch
        model.weighted_sampling_config.update_frequency = 1
    elif cfg.get("load_encoder_from_ckpt", None) is not None:
        log.info(
            f"Loading encoder from checkpoint <{cfg.load_encoder_from_ckpt}>")
        model_old: LightningModule = ContrastiveResNet50.load_from_checkpoint(
            cfg.load_encoder_from_ckpt
        )
        model: LightningModule = hydra.utils.instantiate(
            cfg.module, _recursive_=True
        )
        model.base_encoder = model_old.base_encoder
        model.projection_head = model_old.projection_head

    else:
        # Init lightning model
        log.info(f"Instantiating lightning model <{cfg.module._target_}>")
        model: LightningModule = hydra.utils.instantiate(
            cfg.module, _recursive_=True
        )

    # Init callbacks
    log.info("Instantiating callbacks...")
    callbacks: List[Callback] = instantiate_callbacks(
        cfg.get("callbacks")
    )

    for callback in callbacks:
        if isinstance(callback, LinearWarmupCosineAnnealingLR) and cfg.get("restore_lr_last_epoch", 0) > 0:
            print("restore_lr")
            callback.max_epochs = cfg.get("restore_lr_max_epochs", 350)
            callback.last_epoch = cfg.get("restore_lr_last_epoch", 0)

    # Init loggers
    log.info("Instantiating loggers...")
    logger = instantiate_loggers(
        cfg.get("logger")
    )

    # Init lightning trainer
    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer, callbacks=callbacks, logger=logger
    )

    # Send parameters from cfg to all lightning loggers
    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)
        if cfg.get("log_grads", False):
            logger[0].watch(model, log='all')

    # ----- START OF MODIFICATIONS -----

    # Get the WandB run ID
    wandb_logger = None
    for lg in logger:
        if hasattr(lg, "experiment") and hasattr(lg.experiment, "id"):
            wandb_logger = lg
    if wandb_logger is not None:
        wandb_run_id = wandb_logger.experiment.id
        log.info(f"WandB Run ID: {wandb_run_id}")

        # Find the checkpoint callback
        checkpoint_callback = None
        print(trainer.callbacks)
        for cb in trainer.callbacks:
            # hasattr(cb, 'dirpath'):
            if isinstance(cb, lightning.pytorch.callbacks.ModelCheckpoint):
                checkpoint_callback = cb
                print(f"Checkpoint callback found: {checkpoint_callback}")
                break
        # modify save_dir
        if checkpoint_callback:
            save_dir = checkpoint_callback.dirpath
            # if path doesnt end in a / append one.
            if not save_dir.endswith('/'):
                save_dir += '/'
            now = datetime.datetime.now()
            save_dir_new = os.path.join(
                save_dir, f"{now.year}-{now.month:02d}-{now.day:02d}-{now.hour:02d}-{now.minute:02d}-{now.second:02d}-{wandb_run_id}")
            checkpoint_callback.dirpath = save_dir_new
            log.info(
                f"Modified save directory to: {checkpoint_callback.dirpath}")
    else:
        log.warning(
            "WandB logger not found. Run ID will not be added to save directory.")

    # Train the model
    if cfg.get("train"):

        # if cfg.get("ckpt_path", False):
        #     model = ContrastiveResNet50()
        #     trainer.fit(
        #     model=model,
        #     datamodule=datamodule,
        #     ckpt_path=cfg.get("ckpt_path"),
        #     )
        # else:

        log.info("Starting training!")
        trainer.fit(
            model=model,
            datamodule=datamodule,
            ckpt_path=cfg.get("ckpt_path"),
        )

    train_metrics = trainer.callback_metrics

    # Test the model
    if cfg.get("test"):
        log.info("Starting testing!")
        log.info("test last checkpoint")
        ckpt_path = trainer.checkpoint_callback.last_model_path
        trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)

        ckpt_path = trainer.checkpoint_callback.best_model_path
        if ckpt_path == "":
            log.warning(
                "Best ckpt not found! Using current weights for testing..."
            )
            ckpt_path = None
        trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
        log.info(f"Best ckpt path: {ckpt_path}")

        if cfg.get("test_balanced", False):
            log.info("Starting testing on balanced dataset!")
            datamodule.subsample_balanced_test = True
            datamodule.setup()
            trainer.test(
                model=model,
                datamodule=datamodule,
                ckpt_path=ckpt_path,
            )

    test_metrics = trainer.callback_metrics

    # merge train and test metrics
    metric_dict = {**train_metrics, **test_metrics}

    return metric_dict, object_dict


@hydra.main(**_HYDRA_PARAMS)
def main(cfg: DictConfig) -> Optional[float]:
    # torch.multiprocessing.set_sharing_strategy('file_system')

    print(OmegaConf.to_yaml(cfg, resolve=True))

    # train the model
    metric_dict, _ = train(cfg)

    return metric_dict


if __name__ == "__main__":
    torch.set_float32_matmul_precision('medium')
    main()

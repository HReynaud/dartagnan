from gettext import find
import os
import torch
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader
from pytorch_lightning.plugins import DDPPlugin
import wandb
import logging
from utils.arguments import parse_arguments
from utils.functions import (
    verify_checkpoint_availability,
    get_checkpoint_callback,
    WandbArgsUpdate,
)
from pl_modules.datasets import get_datamodule
from pl_modules.models import get_modelmodule
from utils.constants import (
    WANDB_CACHE_DIR,
    TORCH_HOME,
    WANDB_ENTITY,
    PROJECT_NAME,
)

os.environ["WANDB_CACHE_DIR"] = WANDB_CACHE_DIR
os.environ["TORCH_HOME"] = TORCH_HOME

def main():
    """
    Main function to train all models.
    """
    args = parse_arguments()
    debug = False

    # 0. Set debug mode
    limit_batches = 1.0 if not debug else 0.05
    if debug:
        os.environ["DEBUG_VERBOSE"] = "True"
        print("Only using 5 percent of data in debug mode")

    # 1. Set up wandb
    logger = WandbLogger(
        project=PROJECT_NAME,
        log_model=False,
        name=args.name,
        entity=WANDB_ENTITY,
    )

    # 2. Set up logger level
    logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)

    # 3. Enforce reproducibility, has been proven to not be really deterministic anyway due to some cuda functions
    seed_everything(args.seed, workers=True)

    # 4. Prepare storage path
    checkpoint_path = verify_checkpoint_availability(args)

    # 5. Get data and model wrappers
    datamodule = get_datamodule(args)
    modelmodule = get_modelmodule(args)

    # 6. Set up callbacks
    checkpoint_callback = get_checkpoint_callback(args)
    lr_callback = LearningRateMonitor(logging_interval="epoch")
    callbacks = [
        checkpoint_callback,
        WandbArgsUpdate(args),
        lr_callback,
    ] 

    # 7. Set up trainer
    trainer = Trainer(
        default_root_dir=checkpoint_path,
        callbacks=callbacks,
        logger=logger,
        max_epochs=args.train.epochs,
        deterministic=False,
        limit_train_batches=limit_batches,
        limit_val_batches=limit_batches,
        precision=16,
        accelerator="ddp",
        gpus=args.gpus,
        plugins=DDPPlugin(find_unused_parameters=True)
    )

    # 7. Train model
    trainer.fit(modelmodule, datamodule=datamodule)


if __name__ == "__main__":
    main()
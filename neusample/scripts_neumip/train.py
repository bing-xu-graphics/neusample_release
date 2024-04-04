import argparse
from gc import callbacks
import os
import sys
from unicodedata import name

import torch as th
import pytorch_lightning as pl
from pytorch_lightning.callbacks import RichProgressBar

# Insert root dir in path
p = os.path.abspath('.')
sys.path.insert(1, p)

from neumat import interfaces, config

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", help=".yml config file")
    parser.add_argument("--seed", type=int, default=0, help="random seed for reproducibility")
    parser.add_argument("--accelerator", type=str, default="gpu")
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--checkpoint_dir", help="directory for checkpoints", default="checkpoints")

    args = parser.parse_args()

    conf = config.get_config(config_filename=args.config)

    pl.seed_everything(args.seed)

    model_interface = getattr(interfaces, conf.training_interface)

    # TODO: Add init_from checkpoint support
    model = model_interface(conf, **conf.interface_params)

    data_interface = getattr(interfaces, conf.data_interface)
    data = data_interface(conf, **conf.data_interface_params)

    logger = pl.loggers.TensorBoardLogger(args.checkpoint_dir, name=conf.name)
    cbks = [
        pl.callbacks.RichProgressBar(),
        pl.callbacks.ModelCheckpoint(
            save_last=True
        ),
        pl.callbacks.LearningRateMonitor()
    ]

    trainer = pl.Trainer.from_argparse_args(
        args,
        # limit_train_batches=1,
        default_root_dir=args.checkpoint_dir,
        # deterministic=True, # TODO: grid sampler on cuda isn't deterministic
        accelerator=args.accelerator,
        max_epochs=conf.num_epochs,
        devices=args.num_gpus,
        auto_select_gpus=True,
        logger=logger,
        callbacks=cbks
    )

    trainer.fit(model, data)

if __name__ == "__main__":

    main()
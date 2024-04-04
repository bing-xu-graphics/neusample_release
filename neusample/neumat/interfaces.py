"""Model and data interfaces for training jobs"""

import pytorch_lightning as pl
import torch
import numpy as np
from utils import ops
from torch.utils.data import DataLoader

from . import datasets, models, config

class NeuMIPv1Module(pl.LightningModule):
    """A model for NeuMIP: Multi-resolution neural materials
    Ref: https://cseweb.ucsd.edu/~viscomp/projects/NeuMIP/

    Args:

    """

    def __init__(self,
                 conf: config.BaseNeuMatConfig,
                 lr: float = 1e-3,
                 loss: str = "l1",
                 ):
        super().__init__()

        self.conf = config.merge(config.BaseNeuMatConfig.default_config(), conf)
        self.save_hyperparameters()

        self.lr = lr
        self.loss = loss

        model_cls = getattr(models, conf.model)
        self.model = model_cls(**conf.model_params)

        if loss == "l1":
            self.loss_fn = torch.nn.L1Loss()
        else:
            raise ValueError("Invalid loss function specified")

    def configure_optimizers(self):
        opt_params = self.model.parameters()
        opt = torch.optim.Adam(
            opt_params,
            lr=self.lr
        )
        return opt

    def forward(self, camera_dir, light_dir, uv):
        camera_dir = camera_dir.to(self.device)
        light_dir = light_dir.to(self.device)
        uv = uv.to(self.device)

        rgb_result = self.model(camera_dir, light_dir, uv)
        return rgb_result



    def training_step(self, batch: dict, batch_idx: int):

        self.model.set_iteration_index(self.global_step)

        camera_dir, light_dir, color, uv, cam_qr = batch

        camera_dir = camera_dir.reshape(-1, camera_dir.shape[-1])
        light_dir = light_dir.reshape(-1, light_dir.shape[-1])
        uv = uv.reshape(-1, uv.shape[-1])
        color = color.reshape(-1, color.shape[-1])

        rgb_prediction = self.forward(camera_dir, light_dir, uv)

        loss = self.loss_fn(rgb_prediction, color)

        with torch.no_grad():
            self.log("psnr", ops.psnr(loss.cpu()), prog_bar=True)

        return loss


    def on_fit_start(self):
        self.print(
            f"Training model \"{self.conf.name}\" "
            f"on dataset: {self.conf.training_dataset}")



class NeuMIPv1DataModule(pl.LightningDataModule):
    """Set up the train/validation datasets for NeuMIPv1"""
    def __init__(self,
                 conf: config.BaseNeuMatConfig,
                 bs: int = 1, # Needs to be a perfect square
                 val_bs: int = 1, # Needs to be a perfect square
                 num_workers: int = 0):
        super().__init__()
        self.conf = conf
        self.bs = bs
        self.val_bs = val_bs
        self.num_workers = num_workers

    def setup(self, stage=None):
        self.trainset = datasets.NeuMIPv1Dataset(self.conf.training_dataset)

        self.valset = None
        # TODO: Add validation dataset

    def train_dataloader(self):
        return DataLoader(self.trainset,
                          batch_size=self.bs,
                          num_workers=0,
                          pin_memory=False,
                          shuffle=True,
                          drop_last=True)

    # def val_dataloader(self):
    #     return None

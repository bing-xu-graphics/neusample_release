
import torch
import torch.nn as nn
import torch.nn.functional as NF
import torch.optim as optim
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

import math


import os
from pathlib import Path
from argparse import Namespace, ArgumentParser



from configs.config import default_options
from model.neumip import NeuMIP
from model.mlps import mlp,PositionalEncoding
from model.histogram2d import Histogram2D

from utils.dataset.neumip_dataset import NeuMIPDataset

class ModelTrainer(pl.LightningModule):
    """ diver model training code """
    def __init__(self, hparams: Namespace, *args, **kwargs):
        super(ModelTrainer, self).__init__()
        self.save_hyperparameters(hparams)
        self.lobe_size = hparams.lobe_size
        self.wi_num = hparams.wi_num
        
        self.model = NeuMIP()
        self.model.load_state_dict(
            torch.load(hparams.neumip_path,map_location='cpu'))
        for p in self.model.parameters():
            p.requires_grad = False
        
        layer_num,Ci,C,D,Co,res,encode = hparams.base_mlp
        self.base2d = Histogram2D(32,mode='nearest')
        self.layer_num = layer_num
        self.input_encode = PositionalEncoding(0)
        
    def __repr__(self):
        return repr(self.hparams)

    def configure_optimizers(self):
        if(self.hparams.optimizer == 'SGD'):
            opt = optim.SGD
        if(self.hparams.optimizer == 'Adam'):
            opt = optim.Adam

        optimizer = opt(self.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay)

        scheduler = optim.lr_scheduler.MultiStepLR(optimizer,milestones=self.hparams.milestones,gamma=self.hparams.scheduler_rate)
        return [optimizer], [scheduler]
    
    def train_dataloader(self,):
        spatial,angular = self.hparams.dataset_length
        dataset = NeuMIPDataset(spatial,angular,split='train')
        return DataLoader(dataset, shuffle=True,batch_size=self.hparams.batch_size)
    
    def val_dataloader(self,):
        dataset = NeuMIPDataset(2,3,split='val')
        return DataLoader(dataset, shuffle=False)

    def forward(self, x):
        return
    
    def training_step(self, batch, batch_idx):
        """ one training step
        """
        point = batch['uv']
        wo = batch['wo']
        
        # importance sampling incident direction
        
        B = point.shape[0]
        
        S = 1024
        sy,sx = torch.meshgrid(
            torch.linspace(0,31,32,device=point.device),
            torch.linspace(0,31,32,device=point.device)
        )
        wi = (torch.stack([sx,sy],-1).reshape(1,-1,2) + torch.rand(B,S,2,device=point.device))/32.0
        #lobes = self.model.get_lobe(point,wo,self.lobe_size).mean(-1)
        #lobes = lobes.reshape(B,-1)
        #lobes = lobes/lobes.sum(-1,keepdim=True).clamp_min(1e-12)*self.lobe_size*self.lobe_size
        #idx = (wi*self.lobe_size).long().clamp_max(self.lobe_size-1)
        #pdfs = torch.gather(lobes.reshape(B,-1),1,
        #                    idx[...,1]*self.lobe_size+idx[...,0])
        wi = wi*2-1
        wi_z = (1-wi.pow(2).sum(-1))
        valid = wi_z >=0
        wi = torch.cat([wi,wi_z.relu().sqrt()[...,None]],dim=-1)
        
        btf_ = self.model.get_brdf(point[torch.where(valid)[0]],wo[torch.where(valid)[0]],wi[valid]).mean(-1)
        btf = torch.zeros(B,S,device=btf_.device)
        btf[valid] = btf_*4
        
        _,S,_ = wi.shape
        
        # read condition feature
        rgb_feature = self.model.get_rgb_texture(point,wo)
        condition_feature = torch.cat([rgb_feature,self.input_encode(wo[...,:2])],-1)
        
        ret = self.base2d(wi[...,:2].reshape(B*S,2),condition_feature.repeat_interleave(S,0))
        pdf_est = ret.reshape(B,S)
        loss = NF.mse_loss(btf,pdf_est)

        if loss.isnan() or loss.isinf():
            return None
        self.log('train/loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        """ one validation step
        """

        point = batch['uv']
        wo = batch['wo']
        
        # importance sampling incident direction
        lobes = self.model.get_lobe(point,wo,self.lobe_size).mean(-1)[0]
        lobes = (lobes/lobes.sum().clamp_min(1e-12))
    
    
        # read condition feature
        rgb_feature = self.model.get_rgb_texture(point,wo)
        condition_feature = torch.cat([rgb_feature,self.input_encode(wo[...,:2])],-1)
    
        uv = torch.meshgrid(
            torch.linspace(-1,1,self.lobe_size,device=point.device),
            torch.linspace(-1,1,self.lobe_size,device=point.device)
        )
        uv = torch.stack([uv[1],uv[0]],-1).reshape(1,-1,2)
        w = (1-uv.pow(2).sum(-1))
        valid = w >= 0
        
        wi = uv
        
        lobes_est = torch.zeros_like(lobes).reshape(-1)
        
        batch_size = 1024#0//2
        for b in range(math.ceil(self.lobe_size*self.lobe_size*1.0/batch_size)):
            b0 = b*batch_size
            b1 = min(b0+batch_size,self.lobe_size**2)
            logq = self.base2d(wi[0,b0:b1],condition_feature.repeat_interleave((b1-b0),0))
            lobes_est[b0:b1] = logq/self.lobe_size/self.lobe_size
        lobes_est = lobes_est * valid[0]
        lobes_est = lobes_est/lobes_est.sum().clamp_min(1e-12)
        lobes_est = lobes_est.reshape(self.lobe_size,self.lobe_size)
        
        loss = NF.mse_loss(lobes,lobes_est)/lobes.abs().max().pow(2).clamp_min(1e-12)
        
        psnr = -10.0 * math.log10(loss.clamp_min(1e-10))
        
        
        self.log('val/loss', loss)#, on_step=True)
        self.log('val/psnr', psnr)#, on_step=True)

        lobes = lobes/(lobes.max()+1e-8)
        lobes_est = lobes_est/(lobes_est.max()+1e-8)
        
        self.logger.experiment.add_image('val/gt_image', lobes, batch_idx, dataformats='HW')
        self.logger.experiment.add_image('val/inf_image', lobes_est, batch_idx, dataformats='HW')
        return

            
def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        for name, args in default_options.items():
            if(args['type'] == bool):
                parser.add_argument('--{}'.format(name), type=eval, choices=[True, False], default=str(args.get('default')))
            else:
                parser.add_argument('--{}'.format(name), **args)
        return parser
        
if __name__ == '__main__':

    torch.manual_seed(9)
    torch.cuda.manual_seed(9)

    parser = ArgumentParser()
    parser = add_model_specific_args(parser)
    hparams, _ = parser.parse_known_args()

    # add PROGRAM level args
    parser.add_argument('--experiment_name', type=str, required=True)
    parser.add_argument('--max_epochs', type=int, default=500)
    parser.add_argument('--log_path', type=str, default='./logs')
    parser.add_argument('--ft', type=str, default=None)
    parser.add_argument('--checkpoint_path', type=str, default='./checkpoints')
    parser.add_argument('--resume', dest='resume', action='store_true')
    parser.add_argument('--device', type=int, required=False,default=None)

    parser.set_defaults(resume=False)
    args = parser.parse_args()
    args.gpus = [args.device]
    experiment_name = args.experiment_name

    # setup checkpoint loading
    checkpoint_path = Path(args.checkpoint_path) / experiment_name
    log_path = Path(args.log_path)
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    
    checkpoint_callback = ModelCheckpoint(checkpoint_path, monitor='val/loss', save_top_k=1, save_last=True)
    logger = TensorBoardLogger(log_path, name=experiment_name)

    last_ckpt = checkpoint_path / 'last.ckpt' if args.resume else None
    if (last_ckpt is None) or (not (last_ckpt.exists())):
        last_ckpt = None
    else:
        last_ckpt = str(last_ckpt)
    
    # setup model trainer
    model = ModelTrainer(hparams)
    
    trainer = Trainer.from_argparse_args(
        args,
        resume_from_checkpoint=last_ckpt,
        logger=logger,
        checkpoint_callback=checkpoint_callback,
        flush_logs_every_n_steps=1,
        log_every_n_steps=1,
        max_epochs=args.max_epochs
    )

    trainer.fit(model)

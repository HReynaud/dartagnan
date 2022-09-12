import numpy as np
import sklearn.metrics
import wandb
import torch
import torchvision
import os
from torch import optim
from pytorch_lightning import LightningModule
import torch.nn.functional as F

from utils.constants import TRAIN, VAL, TEST
from models.video_expert import EFExpert
from .helpers import (
    get_loss_fn,
    build_checkpoint,
)

class ExpertTrainer(LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.model = EFExpert(args)
        self.lossf = get_loss_fn(args.train.loss)
        self.best_loss = np.inf
        self.confmat = np.zeros((5, 5))
        self.class_weights = torch.tensor(
            [0.13811455, 0.11557594, 0.06535249, 0.01163264, 0.66932438],
            dtype=torch.float32,
        )
        self.idx_labl = {TRAIN: [], VAL: []}
        self.idx_pred = {TRAIN: [], VAL: []}
        self.last_loss = 0
    
    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        opt = optim.Adam(self.model.get_parameters(), lr=self.args.train.lr)

        if self.args.train.sched == "exp":
            sch = optim.lr_scheduler.ExponentialLR(
                opt, 0.95, last_epoch=-1, verbose=False
            )
            return {"optimizer": opt, "scheduler": sch, "interval": "epoch"}
        return opt
    
    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx, TRAIN)

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx, VAL)

    def training_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, TRAIN)

    def validation_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, VAL)
    
    def shared_epoch_end(self, outputs, name):
        if self.local_rank == 0:
            data = [[x, y] for (x, y) in zip(self.idx_labl[name], self.idx_pred[name])]
            table = wandb.Table(data=data, columns=["Label", "Prediction"])
            self.logger.experiment.log(
                {
                    f"{name}/scatter": wandb.plot.scatter(
                        table, "Label", "Prediction", title="Label vs Prediction"
                    ),
                    f"{name}/R2": sklearn.metrics.r2_score(
                        self.idx_labl[name], self.idx_pred[name]
                    ),
                    f"{name}/MAE": sklearn.metrics.mean_absolute_error(
                        self.idx_labl[name], self.idx_pred[name]
                    ),
                    f"{name}/MSE": sklearn.metrics.mean_squared_error(
                        self.idx_labl[name], self.idx_pred[name]
                    ),
                    f"{name}/loss": self.last_loss, #see shared_step
                }
            )
            self.idx_labl[name] = []
            self.idx_pred[name] = []
        return None

    def shared_step(self, batch, batch_idx, name):
        # videos, lvefs, classes, emb, _ 
        _, videos, lvefs, _, _ = batch


        pred_lvef = self.model(videos.float()).view(-1)

        loss = torch.nn.functional.mse_loss(pred_lvef, lvefs.half())

        self.idx_labl[name].extend(lvefs.cpu().detach().tolist())
        self.idx_pred[name].extend(pred_lvef.cpu().detach().tolist())

        # print(batch_idx, self.args.wandb.interval, batch_idx % self.args.wandb.interval == 0,  self.local_rank)
        if batch_idx % self.args.wandb.interval == 0 and self.local_rank == 0:
            log_dict = {
                f"{name}/loss": loss.item(),
            }
            self.logger.experiment.log(log_dict)
            self.last_loss = loss.item()
            # print(log_dict)
        if (
            batch_idx == 0 and 
            self.local_rank == 0 and  
            name == VAL and
            self.args.save_model
        ):
            self.save_model(
                os.path.join(self.args.checkpoint, self.args.name),
                loss,
                self.current_epoch,
            )

        return loss

    def save_model(self, path, loss=np.inf, epoch=-1):
        os.makedirs(path, exist_ok=True)
        data = build_checkpoint(
            models = {"expert_weights": self.model.state_dict()},
            args = self.args,
            epoch = epoch, 
            reference_metric = loss,
            other = {"expert_name": self.model.__class__.__name__},
        )

        torch.save(data, os.path.join(path, f"epoch_{str(epoch).zfill(4)}.pt"))
        if loss < self.best_loss:
            self.best_loss = loss
            torch.save(data, os.path.join(path, "best.pt"))
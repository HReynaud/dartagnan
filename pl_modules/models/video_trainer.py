import numpy as np
import sklearn
import wandb
import torch
import torchvision
import os
from torch import optim
from pytorch_lightning import LightningModule
import torch.nn.functional as F


from utils.constants import TRAIN, VAL, TEST
from models.video_twinnet import VideoTwinNet
from models.video_discriminator import VideoDiscriminator
from .helpers import get_trained_expert
from .helpers import (
    get_random_different_lvef,
    build_checkpoint
)

class VideoTrainer(LightningModule):
    def __init__(self, args) -> None:
        super().__init__()
        self.args = args

        self.model = VideoTwinNet(args)
        self.frame_discriminator = VideoDiscriminator(args)
        self.lvef_discriminator = get_trained_expert(args).eval().to(self.device)

        self.metrics = {
            "D_real": [],
            "D_fake": [],
            "Rec": [],
            "EF": [],
            "GAN": [],
        }
        self.best_loss = np.inf
        self.bypass_discriminator = False
        self.bypass_generator = False
        self.lossf = torch.nn.L1Loss()
        self.automatic_optimization = False
        self.epoch_loss = []

    def configure_optimizers(self):
        g_opt = torch.optim.Adam(self.model.get_parameters(), lr=self.args.train.lr_generator)
        d_opt = torch.optim.Adam(
            self.frame_discriminator.get_parameters(), lr=self.args.train.lr_disc
        )
        # By default, divides by 10 over 25 epochs with exponential decay
        gamma = (1/self.args.train.max_lr_decay)**(1/self.args.train.decay_epochs)  
        g_scheduler = optim.lr_scheduler.ExponentialLR(
            g_opt, gamma, verbose=False
        )
        d_scheduler = optim.lr_scheduler.ExponentialLR(
            d_opt, gamma, verbose=False
        )

        return [g_opt, d_opt], [g_scheduler, d_scheduler]

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx, TRAIN)

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx, VAL)

    def shared_step(self, batch, batch_idx, name):

        # Retrieve optimizers
        g_opt, d_opt = self.optimizers()

        # Create labels on current device
        true_label = torch.ones((1,), dtype=torch.half, device=self.device)
        false_label = torch.zeros((1,), dtype=torch.half, device=self.device)

        # Get loss weights
        weight_disc = self.args.train.disc_weight * (
            torch.sigmoid(torch.tensor(self.current_epoch - self.args.train.disc_weight_offset))
            if self.args.train.disc_weight_offset > 0
            else 1
        )
        weight_expert = self.args.train.expert_weight * (
            torch.sigmoid(torch.tensor(self.current_epoch - self.args.train.expert_weight_offset))
            if self.args.train.expert_weight_offset > 0
            else 1
        )
        weight_recon = self.args.train.recon_weight * (
            torch.sigmoid(torch.tensor(self.current_epoch - self.args.train.recon_weight_offset))
            if self.args.train.recon_weight_offset > 0
            else 1
        )

        # Get inputs and labels
        filenames, videos, lvefs, es_index, ed_index = batch

        # Z = videos.to(torch.float32)
        # X = lvefs.to(torch.float32)/100.0

        Z = videos.half()  # BxCxFxHxW | 0-1
        X = lvefs.half() / 100.0  # B  | 0-100 -> 0-1

        X_s = get_random_different_lvef(lvefs, margin=10, global_clip=None).half() / 100.0 # B | 0-100 -> 0-1

        # Forward pass
        Y_pred, Y_s_pred = self.model(X, X_s, Z)  # BxCxFxHxW | 0-1
        lvef_pred = self.lvef_discriminator.eval()(Y_s_pred) / 100.0  # B | 0-100 -> 0-1

        # Train discriminator on real data
        pred_real = self.frame_discriminator(Z)
        loss_dr = self.lossf(pred_real, true_label.expand_as(pred_real))
        if self.training and not self.bypass_discriminator:
            d_opt.zero_grad()
            self.manual_backward(loss_dr)
            d_opt.step()

        # Train discriminator on fake data
        pred_fake = self.frame_discriminator(Y_s_pred.detach())
        loss_df = self.lossf(pred_fake, false_label.expand_as(pred_fake))
        if self.training and not self.bypass_discriminator:
            d_opt.zero_grad()
            self.manual_backward(loss_df)
            d_opt.step()

        # Train generator
        pred = self.frame_discriminator(Y_s_pred)
        error_disc = self.lossf(pred, true_label.expand_as(pred)) * weight_disc
        error_recon = self.lossf(Y_pred, Z) * weight_recon
        error_expert = self.lossf(lvef_pred.view(-1), X_s.view(-1)) * weight_expert
        loss_g = error_disc + error_recon + error_expert
        if self.training and not self.bypass_generator:
            g_opt.zero_grad()
            self.manual_backward(loss_g)
            g_opt.step()

        # Log metrics
        self.metrics["D_real"].append(loss_dr.item())
        self.metrics["D_fake"].append(loss_df.item())
        self.metrics["Rec"].append(error_recon.item())
        self.metrics["EF"].append(error_expert.item())
        self.metrics["GAN"].append(error_disc.item())

        # Log metrics to wandb
        if (
            self.args.wandb.enabled # check if wandb is enabled
            and batch_idx % self.args.wandb.interval == 0 # check if it is time to log
            and self.local_rank == 0 # check if it is the main process
            and len(self.metrics["GAN"]) > 0
        ):

            self.log_to_wandb(
                name,
                X, X_s, Z, Y_pred, Y_s_pred, lvef_pred, 
                weight_disc, weight_expert, weight_recon, 
                g_opt, d_opt
            )
            
            self.metrics["D_real"] = []
            self.metrics["D_fake"] = []
            self.metrics["Rec"] = []
            self.metrics["EF"] = []
            self.metrics["GAN"] = []            

        if (
            batch_idx == 0
            and self.local_rank == 0
            and name == VAL
            and self.args.save_model
        ):  

            self.save_model(
                os.path.join(self.args.checkpoint, self.args.name),
                loss_g,
                self.current_epoch,
            )
            self.epoch_loss = []
        return None

    def save_model(self, path, loss=np.inf, epoch=-1):
        os.makedirs(path, exist_ok=True)
        data = build_checkpoint(
            models={
                "generator": self.model.state_dict(),
                "frame_discriminator": self.frame_discriminator.state_dict(),
                "lvef_discriminator": self.lvef_discriminator.state_dict(),
            },
            args=self.args,
            epoch = epoch, 
            reference_metric = loss,
        )
        torch.save(data, os.path.join(path, f"epoch_{str(epoch).zfill(4)}.pt"))
        if loss < self.best_loss:
            self.best_loss = loss
            torch.save(data, os.path.join(path, "best.pt"))

    def training_epoch_end(self, outputs) -> None:
        if self.args.train.sched == "exp" and self.current_epoch < self.args.train.max_lr_decay:
            g_sch, d_sch = self.lr_schedulers()
            g_sch.step()
            d_sch.step()
    
    def log_to_wandb(self, name, X, X_s, Z, Y_pred, Y_s_pred, lvef_pred, weight_disc, weight_expert, weight_recon, g_opt, d_opt):
        wb_X = X[0].cpu().detach().numpy().item()
        wb_X_s = X_s[0].cpu().detach().numpy().item()
        wb_Z = wandb.Video(
            np.clip((Z[0].permute(1, 0, 2, 3).repeat(1, 3, 1, 1).cpu().detach().numpy() * 255), 0, 255).astype(np.uint8),
            fps=8,
            format="gif",
            caption=f"{name}_input",
        )
        wb_Y_pred = wandb.Video(
            np.clip((Y_pred[0].permute(1, 0, 2, 3).repeat(1, 3, 1, 1).cpu().detach().numpy() * 255), 0, 255).astype(np.uint8),
            fps=8,
            format="gif",
            caption=f"{name}_reconstructed",
        )
        wb_Y_s_pred = wandb.Video(
            np.clip((Y_s_pred[0].permute(1, 0, 2, 3).repeat(1, 3, 1, 1).cpu().detach().numpy()* 255), 0, 255).astype(np.uint8),
            fps=8,
            format="gif",
            caption=f"{name}_generated",
        )
        with torch.no_grad():
            wb_X_pred = (self.lvef_discriminator(Y_pred[0:1].float()).cpu().detach().numpy().item())
            wb_X_s_pred = lvef_pred[0].cpu().detach().numpy().item()

            wbtable = wandb.Table(
                columns = ["Branch", "X", "Z", "Y", "Pred LVEF"],
                data = [
                    ["Factual", wb_X, wb_Z, wb_Y_pred, wb_X_pred],
                    ["Counterfactual", wb_X_s, wb_Z, wb_Y_s_pred, wb_X_s_pred]
                    ]
            )

            loss = (
                sum(self.metrics["GAN"])
                + sum(self.metrics["Rec"])
                + sum(self.metrics["EF"])
                + sum(self.metrics["D_real"])
                + sum(self.metrics["D_fake"])
            ) / len(self.metrics["GAN"])

            self.logger.experiment.log(
                {
                    f"{name}/Loss D_real": sum(self.metrics["D_real"])
                    / len(self.metrics["D_real"]),
                    f"{name}/Loss D_fake": sum(self.metrics["D_fake"])
                    / len(self.metrics["D_real"]),
                    f"{name}/Loss Rec": sum(self.metrics["Rec"])
                    / len(self.metrics["D_real"]),
                    f"{name}/Loss EF": sum(self.metrics["EF"])
                    / len(self.metrics["D_real"]),
                    f"{name}/Loss GAN": sum(self.metrics["GAN"])
                    / len(self.metrics["D_real"]),
                    f"{name}/loss": loss,
                    f"{name}/Weight EF": weight_expert,
                    f"{name}/Weight Disc": weight_disc,
                    f"{name}/Weight Rec": weight_recon,
                    f"{name}/table": wbtable,
                    f"{name}/epoch": self.current_epoch,
                    f"{name}/G_lr": g_opt.param_groups[0]["lr"],
                    f"{name}/D_lr": d_opt.param_groups[0]["lr"],
                }
            )






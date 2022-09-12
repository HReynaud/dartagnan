import wandb
import numpy as np
import torch
import torch.nn as nn
from argparse import Namespace

def get_trained_expert(args):
    """
    Build a trained expert model from a checkpoint.
    """
    import models.video_expert as expert_model_collection

    model_path = args.train.trained_expert_path
    expert_dict = torch.load(model_path)

    model_args = expert_dict["args"]

    model_name = expert_dict["expert_name"]
    assert model_name in expert_model_collection.__dict__
    assert hasattr(expert_model_collection.__dict__[model_name], 'with_weights')

    model = expert_model_collection.__dict__[model_name](model_args)

    model_weights = expert_dict["expert_weights"]
    model.load_state_dict(model_weights)
    model = model.to(torch.device('cpu'))

    return model

def get_random_different_lvef(lvef, margin=10, global_clip=None):
    lvef[lvef <= margin] = margin + 0.5
    lvef[lvef >= 100 - margin] = 100 - margin - 0.5

    lvef = lvef[:, None]

    plage = torch.arange(0, 101, device=lvef.device).repeat(lvef.shape[0], 1)

    plage = plage[(plage >= lvef + margin).logical_or(plage <= lvef - margin)].reshape(
        lvef.shape[0], -1
    )
    return plage[
        torch.arange(lvef.shape[0]), torch.randint(0, plage.shape[1], (lvef.shape[0],))
    ]

def get_loss_fn(name):
    if name == "l1":
        return nn.L1Loss()
    elif name == "l2":
        return nn.MSELoss()
    elif name == "smooth_l1":
        return nn.SmoothL1Loss()
    elif name == "bce":
        return nn.BCELoss()
    elif name == "bce_logits":
        return nn.BCEWithLogitsLoss()
    elif name == "ce":
        return nn.CrossEntropyLoss()
    elif name == "mse":
        return nn.MSELoss()
    elif name == "kldiv":
        return nn.KLDivLoss()
    elif name == "nll":
        return nn.NLLLoss()
    elif name == "poisson":
        return nn.PoissonNLLLoss()
    elif name == "cosine":
        return nn.CosineEmbeddingLoss()
    elif name == "hinge":
        return nn.HingeEmbeddingLoss()
    elif name == "multi_margin":
        return nn.MultiMarginLoss()
    elif name == "multi_label":
        return nn.MultiLabelMarginLoss()
    elif name == "smooth_l1_loss":
        return nn.SmoothL1Loss()
    elif name == "soft_margin":
        return nn.SoftMarginLoss()

def build_checkpoint( 
    models:dict[torch.nn.Module], 
    args:Namespace, 
    epoch:int, 
    reference_metric:float, 
    other:dict=None) -> dict:
    
    ckpt = {}
    for k,v in models.items():
        ckpt[k] = v
    ckpt["args"] = args
    ckpt["epoch"] = epoch
    ckpt["reference_metric"] = reference_metric
    if other:
        for k,v in other.items():
            ckpt[k] = v
    return ckpt
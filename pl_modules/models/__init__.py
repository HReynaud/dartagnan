import torch

from .video_trainer import VideoTrainer
from .expert_trainer import ExpertTrainer

def get_modelmodule(args):
    import pl_modules.models as modules

    name = slug = args.train.model
    
    if "Trainer" in name:
        slug = (name.split("Trainer")[0] + "_Trainer").lower()
    
    if not(slug in modules.__dict__):
        raise ValueError(
            f"Datamodule '{name}' not found, available modules are:", 
            [k for k in modules.__dict__ if "trainer" in k],
        )
    else:
        return getattr(modules, slug).__dict__[name](args)


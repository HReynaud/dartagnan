"""
Find if a dataset with the specified name exists.
If it does, transform it into a Pytorch Lightning DataModule and return it.
Otherwise, raise an error and list all possible datasets.
"""

from torch.utils.data import Dataset
from .dm_builder import DataModuleWrapper


def get_datamodule(args):
    import datasets as modules

    name = args.dataset.dataset
    
    if not(name.lower() in modules.__dict__):
        available_datasets = []
        for k in modules.__dict__:
            if getattr(modules, k.lower()).__class__.__name__ == "module" and \
                k in getattr(modules, k.lower()).__dict__:
                available_datasets.append(k)
        raise ValueError(
            f"Datamodule '{name}' not found, available modules are:",
            available_datasets,
        )
    else:
        return DataModuleWrapper(args, getattr(modules, name.lower()).__dict__[name])

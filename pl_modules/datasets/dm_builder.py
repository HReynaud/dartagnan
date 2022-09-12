from argparse import Namespace
import os
from torch.utils.data import DataLoader, Dataset
from pytorch_lightning import LightningDataModule

class DataModuleWrapper(LightningDataModule):
    def __init__(self, args: Namespace, ds_class: Dataset):
        super(DataModuleWrapper, self).__init__()
        self.args = args
        self.dataset = {}
        self.ds_class = ds_class

    def setup(self, stage=None):
        self.dataset["train"] = self.ds_class(self.args, split="train")
        self.dataset["val"] = self.ds_class(self.args, split="val")
        self.dataset["test"] = self.ds_class(self.args, split="test")

    def train_dataloader(self):
        return DataLoader(
            self.dataset["train"],
            batch_size=self.args.dataset.batch_size,
            num_workers=min(os.cpu_count(), self.args.dataset.batch_size),
            shuffle=True,
            pin_memory=True,
            drop_last=False,
        )

    def val_dataloader(self):
        return DataLoader(
            self.dataset["val"],
            batch_size=self.args.dataset.batch_size,
            num_workers=min(os.cpu_count(), self.args.dataset.batch_size),
            shuffle=False,
            pin_memory=True,
            drop_last=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.dataset["test"],
            batch_size=self.args.dataset.batch_size,
            num_workers=min(os.cpu_count(), self.args.dataset.batch_size),
            shuffle=False,
            pin_memory=True,
            drop_last=False,
        )

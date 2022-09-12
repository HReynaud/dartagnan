import os
import sys
import shutil
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from .constants import VAL

def verbose(*args, **kwargs):
    if os.environ.get("DEBUG_VERBOSE") == "True":
        print(*args, **kwargs)

def query_yes_no(question, default="yes"):
    """Ask a yes/no question via raw_input() and return their answer.

    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
            It must be "yes" (the default), "no" or None (meaning
            an answer is required of the user).

    The "answer" return value is True for "yes" or False for "no".
    """
    valid = {"yes": True, "y": True, "ye": True, "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        sys.stdout.write(question + prompt)
        choice = input().lower()
        if default is not None and choice == "":
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' " "(or 'y' or 'n').\n")

def verify_checkpoint_availability(args):
    verbose("Verifying checkpoint availability...")
    checkpoint_path = os.path.join(args.checkpoint, args.name)
    if os.path.exists(checkpoint_path):
        print("Previous checkpoints for", args.name, "will be deleted.")
        if (query_yes_no("Do you want to delete this folder?", default="no") == False ):
            print("Then change the '--name' argument.")
            exit()
        else:
            shutil.rmtree(checkpoint_path)

    return checkpoint_path

def get_checkpoint_callback(args):
    checkpoint_callback = ModelCheckpoint(
        monitor= f"{VAL}/loss",
        dirpath=os.path.join(args.checkpoint, args.name),
        filename=args.name,
        mode="min",
    )
    return checkpoint_callback

class WandbArgsUpdate(pl.Callback):
    def __init__(self, args) -> None:
        super().__init__()
        self.args = args

    def on_fit_start(self, trainer, pl_module):
        if getattr(trainer.logger.experiment, "dataset", False):
            pass  # not in the logging rank
        else:
            trainer.logger.experiment.config.update(self.args)



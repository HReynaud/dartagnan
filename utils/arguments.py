import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="DARTAGNAN: Deep ARtificial Twin-Architecture GeNerAtive Networks"
    )

    parser, datst_args = parse_dataset(parser)
    parser, model_args = parse_model(parser)
    parser, train_args = parse_train(parser)
    parser, wandb_args = parse_wandb(parser)

    parser.add_argument(
        "--name",
        type=str,
        default="default",
        help="Name of the experiment"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="./checkpoints",
        help="Path to the checkpoint folder"
    )
    parser.add_argument(
        "--save_model",
        type=bool,
        default=True,
        help="Whether to save the model or not"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for entire experiment"
    )
    parser.add_argument(
        "--gpus", 
        type=str, 
        default=-1, 
        help="List of GPU IDs, ex: 0 or 0,1,2"
    )
    
    args = parser.parse_args()

    root_names = {name: value for (name, value) in args._get_kwargs() 
        if name not in datst_args+model_args+train_args+wandb_args}

    datst_names = {name: value for (name, value) in args._get_kwargs() if name in datst_args}
    model_names = {name: value for (name, value) in args._get_kwargs() if name in model_args}
    train_names = {name: value for (name, value) in args._get_kwargs() if name in train_args}
    wandb_names = {name: value for (name, value) in args._get_kwargs() if name in wandb_args}

    root_names['dataset'] = argparse.Namespace(**datst_names)
    root_names['model']   = argparse.Namespace(**model_names)
    root_names['train'] = argparse.Namespace(**train_names)
    root_names['wandb'] = argparse.Namespace(**wandb_names)
    root_namespace = argparse.Namespace(**root_names)

    return root_namespace

def parse_dataset(parser):
    dataset_parser = parser.add_argument_group('dataset', 'Parameters for dataset')
    dataset_parser.add_argument(
        '-d', '--dataset',
        type=str,
        help='Dataset name.',
        default='Echo2s',
    )
    dataset_parser.add_argument(
        '--root',
        type=str,
        help='Path to the dataset folder.',
        default='./data',
    )

    dataset_parser.add_argument(
        '--video_downscale',
        type=int,
        help='Downscale video by this factor.',
        default=1,
    )
    dataset_parser.add_argument(
        '--video_step',
        type=int,
        help='Sample every n frames.',
        default=1,
    )
    dataset_parser.add_argument(
        "--batch_size",
        default=16,
        type=int,
        help="input batch size for training (default: 4)",
    )

    return parser, [action.dest for action in dataset_parser._group_actions]

def parse_model(parser):
    model_parser = parser.add_argument_group('model', 'Parameters for dataset')
    model_parser.add_argument(
        "--noise_injection",
        type=str,
        default="mult",
        choices=["mult", "add", "concat"],
        help="Method for adding noise to the input",
    )
    model_parser.add_argument(
        "--internal_dim",
        type=int,
        default=128,
        help="Internal dim in twin net",
    )    
    model_parser.add_argument(
        "--channels",
        type=int,
        default=1,
        help="Number of channels in the input and output videos",
    )
    model_parser.add_argument(
        "--discriminator_3D",
        action="store_true",
        default=False,
        help="Use 3D discriminator",
    )
    model_parser.add_argument(
        "--share_weights",
        action="store_true",
        default=False,
        help="For Saving the current Model",
    )

    return parser, [action.dest for action in model_parser._group_actions]

def parse_train(parser):
    train_parser = parser.add_argument_group('train', 'Parameters for training')
    train_parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="number of epochs"
    )
    train_parser.add_argument(
        "--lr_generator",
        type=float,
        default=1e-4,
        help="Learning rate for the generator"
    )
    train_parser.add_argument(
        "--lr_disc",
        type=float,
        default=1e-4,
        help="Learning rate for the discriminator"
    )
    train_parser.add_argument(
        "--max_lr_decay",
        type=float,
        default=10.0,
        help="By how much the learning rate is can be devided at most"
    )
    train_parser.add_argument(
        "--decay_epochs",
        type=int,
        default=25,
        help="Over how many epochs the learning rate is decayed to reach the division by max_lr_decay"
    )
    train_parser.add_argument(
        "--disc_weight_offset",
        type=int,
        default=3,
        help="Over how many epochs the discriminator loss grows to reach disc_weight"
    )
    train_parser.add_argument(
        "--disc_weight",
        type=float,
        default=1.0,
        help="Weight of the discriminator loss"
    )
    train_parser.add_argument(
        "--expert_weight_offset",
        type=int,
        default=3,
        help="Over how many epochs the expert loss grows to reach expert_weight"
    )
    train_parser.add_argument(
        "--expert_weight",
        type=float,
        default=1.0,
        help="Weight of the expert loss"
    )
    train_parser.add_argument(
        "--recon_weight_offset",
        type=int,
        default=3,
        help="Over how many epochs the reconstruction loss grows to reach recon_weight"
    )
    train_parser.add_argument(
        "--recon_weight",
        type=float,
        default=1.0,
        help="Weight of the reconstruction loss"
    )
    train_parser.add_argument(
        "--sched",
        type=str,
        default="exp",
        help="Name of the scheduler to use. Options: exp, none"
    )
    train_parser.add_argument(
        "--trained_expert_path",
        type=str,
        default="./checkpoints/expert/expert2.pt",
        help="Path to the expert model weights"
    )
    train_parser.add_argument(
        "--model",
        type=str,
        default="VideoTrainer", #expert_trainer, video_trainer
        help="Trainer to use. Options: expert_trainer, video_trainer"
    )
    train_parser.add_argument(
        "--loss",
        type=str,
        default="ce", #expert_trainer, video_trainer
        help="Loss to use. Options: ce, l1..."
    )
    train_parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate for the expert network training"
    )

    
    return parser, [action.dest for action in train_parser._group_actions]

def parse_wandb(parser):
    wandb_parser = parser.add_argument_group('wandb', 'Parameters for wandb logging')
    wandb_parser.add_argument(
        "--enabled",
        type=bool,
        default=True,
        help="Activate wandb"
    )
    wandb_parser.add_argument(
        "--interval",
        type=int,
        default=10,
        help="Steps between logging, depends on batch size, number of gpus etc"
    )

    return parser, [action.dest for action in wandb_parser._group_actions]


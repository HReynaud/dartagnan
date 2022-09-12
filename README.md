# DARTAGNAN

This repository contains the code for the paper ["D'ARTAGNAN: Counterfactual Video Generation"](https://arxiv.org/abs/2206.01651)


## Presentation - WIP

1) expert models + weights
2) dartagnan + weights
3) results

## Run the code

### 1) Environment
You will need to have conda installed on your machine. Then, create a new environment with the following command:

```bash
conda env create -f environment.yml
```

The versions of the packages are the ones we used for the experiments. If you want to use different versions, you can do so by editing the `environment.yml` file. Be warned that many packages may be incompatible with each other depending on the versions you choose.

Once the environment is created, activate it with the following command:

```bash
conda activate dartagnan
```

Then, install the `dartagnan` package with the following command:

```bash
pip install -e .
```

Finally, configure the `utils/constants.py` file with your `wandb` credentials and project details to track the experiments.


### 2) Data

<ins>Download the data<ins>

The dataset we use is the Echonet-Dynamic dataset, which can be downloaded from [here](https://echonet.github.io/dynamic/index.html#dataset). You will need to request access.

Once downloaded, you need to extract the files in the `data` folder of this repository. The folder structure should look like this:
```
data
├── EchoNet-Dynamic
│   ├── Videos
│   │   ├── 0X1A0A263B22CCD966.avi
│   │   ├── ...
│   ├── FileList.csv
│   ├── VolumeTracings.csv
```

<ins>Process the data<ins>

The data needs to be processed before it can be used. To do so, run the following command:

```bash
# Assuming you are in the root of the repository 
# and you have activated the dartagnan environment
python tools/build_echo_dataset.py -i data/EchoNet-Dynamic -o data/processed_echonet
```

After running this command, the folder structure should look like this:
```
data
├── EchoNet-Dynamic
│   ├── Videos
│   │   ├── 0X1A0A263B22CCD966.avi
│   │   ├── ...
│   ├── FileList.csv
│   ├── VolumeTracings.csv
├── processed_echonet
│   ├── videos
│   │   ├── 0X1A0A263B22CCD966.npy
│   │   ├── ...
│   ├── metainfo.csv
```

### 2) Train the models

<ins>Expert model<ins>

D'ARTAGNAN relies on an Ejection Fraction Regression model, refered to as the `expert model` which has to be pre-trained before we train the counterfactual model. To train the expert model, run the following command:

```bash
python train.py --model ExpertTrainer --root data/echonet --batch_size 4 --name expert --internal_dim 32
```

This will train the expert model and save the weights in the `checkpoints/expert` folder. You can monitor the training with `wandb`. Feel free to play with the hyperparameters to get the best results with the minimal memory footprint.

<ins>D'ARTAGNAN - Counterfactual video generation model<ins>

To train the counterfactual video generation model, run the following command:

```bash
python train.py --model VideoTrainer --root data/processed_echonet --batch_size 2 --name dartagnan --internal_dim 32 --trained_expert_path checkpoints/expert/best.pt
```

This will train the counterfactual video generation model and save the weights in the `checkpoints/dartagnan` folder. You can monitor the training with `wandb`. There are many hyperparameters that you can play with to adjust the memory footprint and the performance of the model.

### 3) Inference

To generate counterfactual videos using a D'ARTAGNAN trained model, run the following command:

WIP

```bash
python tools/inference.py --model_path checkpoints/dartagnan/best.pt --video_path data/EchoNet-Dynamic/Videos/0X1A0A263B22CCD966.avi --output_path results/0X1A0A263B22CCD966.avi
```



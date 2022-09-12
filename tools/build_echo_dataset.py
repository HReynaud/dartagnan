"""
This file requires the echonet dataset, which can be downloaded from:
https://echonet.github.io/dynamic/index.html#dataset
or from:
https://stanfordaimi.azurewebsites.net/datasets/834e1cd1-92f7-4268-9daa-d359198b310a

The echonet datasset will be processed to generate videos of 2 seconds where possible.
Videos which are too short will be skipped.
A new meta information file (FileList.csv in echonet) will be generated which contains the
location of the new ES and ED frames for the newly generated videos.
"""

import os
import pandas as pd
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from utils.video_helpers import loadvideo
from scipy.ndimage import zoom
import numpy as np
import argparse
import shutil


NUMBER_OF_FRAMES = 64
TIME_SPAN = 2 # in seconds


def get_ES_ED_from_labels(labels_path, filenames):
    labels = pd.read_csv(labels_path)
    ES_index = []
    ED_index = []
    for f in tqdm(filenames):
        try:
            ES_index.append(labels[labels["FileName"] == f+".avi"]["Frame"].to_list()[0])
            ED_index.append(labels[labels["FileName"] == f+".avi"]["Frame"].to_list()[-1])
        except:
            ES_index.append(0)
            ED_index.append(0)
    return ES_index, ED_index

def check_and_build_path(path):
    if os.path.exists(path):
        if os.path.isdir(path):
            decision = input(f"{path} already exists. Overwrite(o) / Delete (d) / Cancel (c)? ").lower()
            if decision == 'o':
                print("Existing files will be overwritten.")
            elif decision == 'd':
                print(f"Deleting all existing files in {path}. This may take a while.")
                shutil.rmtree(path)
            elif decision == 'c':
                print("Exiting...")
                exit()
        else:
            raise ValueError(f"{path} is not a directory.")
           
    os.makedirs(os.path.join(path, "videos"), exist_ok=True)
    
    return None


def build_echo_dataset(args):
    print("Building EchoNet dataset...")
    folder_path = os.path.abspath(args.input)
    output_path = os.path.abspath(args.output)

    # Build file paths
    meta_path   = os.path.join(folder_path, "FileList.csv")
    videos_path = os.path.join(folder_path, "Videos")
    labels_path = os.path.join(folder_path, "VolumeTracings.csv")
    out_meta_path = os.path.join(output_path, "metainfo.csv")
    out_videos_path = os.path.join(output_path, "videos")
    check_and_build_path(output_path)
    print("Output path:", output_path)

    # Load metadata
    meta = pd.read_csv(meta_path)
    filenames = meta["FileName"].to_list()
    print(f"Video folder contains {len(os.listdir(videos_path))} elements.")
    print(f"CSV file contains {len(filenames)} elements.")
    print("This process will take around 2h30.")
    # Get ES and ED indices
    # Check if ES and ED indices are in the input file
    if os.path.exists(out_meta_path) and  "ES" in pd.read_csv(out_meta_path).columns.tolist():
        print("Using previous metainfo file.")
        meta = pd.read_csv(out_meta_path)
    # If they are not, retrieve them from the labels file
    else:
        print("Retrieving frame indices from label file:")
        ES_index, ED_index = get_ES_ED_from_labels(labels_path, filenames)
        meta["ES"] = ES_index
        meta["ED"] = ED_index
        meta.to_csv(out_meta_path)

    target_num_images = NUMBER_OF_FRAMES #(2*32 images per second)
    new_ES = []
    new_ED = []

    print("Resampling all videos to 32 fps and 2 seconds...")
    rows_to_drop = []
    for index, row in tqdm(meta.iterrows(), total=len(meta)):
        # Get index of frame between ES and ED frames
        middle = int((row["ES"] + row["ED"]) / 2.0)
        end = row["NumberOfFrames"]-1

        num_images = TIME_SPAN * row["FPS"] # 2 * 1s

        # If the video is too short, skip it
        if abs(row["ES"] - row["ED"]) > num_images:
            print("{} is not a valid video, it will be discarded.".format(row["FileName"]))
            rows_to_drop.append(index)
            continue
        
        # If the video is too long, crop it
        tmp_si = int(middle-(num_images/2.0))
        start_index = max(tmp_si, 0)
        oveflows = abs(tmp_si) if tmp_si < 0 else 0

        tmp_ei = int(middle+(num_images/2.0))
        end_index = min(tmp_ei, end)
        overflowe = abs(end-tmp_ei) if tmp_ei > end else 0

        start_index = int(start_index-overflowe)
        end_index = int(end_index+oveflows)

        # Compute the new ES and ED indices
        new_ES.append(int(np.rint((row["ES"]-start_index) / (end_index-start_index) * target_num_images)))
        new_ED.append(int(np.rint((row["ED"]-start_index) / (end_index-start_index) * target_num_images)))

        # Debugging :
        if end_index-start_index < 1:
            print("{} is not a valid video, it will be discarded.".format(row["FileName"]))
            rows_to_drop.append(index)

        # Apply crop to video and save it
        video_path = os.path.join(videos_path, row["FileName"]+'.avi')
        video = loadvideo(video_path)
        video = video[:,start_index:end_index].mean(axis=0)
        if video.shape[0] < 1:
            print("{} is not a valid video, it will be discarded.".format(row["FileName"]))
            rows_to_drop.append(index)
            new_ES = new_ES[:-1]
            new_ED = new_ED[:-1]
        else:
            zoomf = target_num_images/video.shape[0]
            # video = np.clip(zoom(video, [zoomf, 1, 1]), 0, 255).astype(np.uint8)
            # if video.shape != (target_num_images, 112, 112):
            #     print(row["FileName"], video.shape)
            #     print(video.dtype, video.min(), video.max())
            # np.save(os.path.join(out_videos_path, row["FileName"]+'.npy'), video)
    
    # Remove rows that are not valid videos (too short)
    print(rows_to_drop)
    meta.drop(rows_to_drop, inplace=True)
    print(f"{len(rows_to_drop)} rows have been dropped.")

    # Save new ES and ED indices into meta file
    meta["2S_ES"] = new_ES
    meta["2S_ED"] = new_ED

    print(f"Saving new metainfo file to {out_meta_path}.")
    meta.to_csv(out_meta_path)

def parse_args():
    parser = argparse.ArgumentParser(description='Build EchoNet dataset.')
    parser.add_argument(
        '-i', '--input', 
        type=str,
        help='Path to the root folder of the EchoNet dataset.')
    parser.add_argument(
        '-o', '--output',
        type=str,
        help='Path to the dataset output folder.')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    build_echo_dataset(args)
import os

from pathlib import Path

import numpy as np
import pandas as pd

from tqdm import tqdm


def one_hot_vector(idx: int, num_classes: int) -> np.array:
    """
    create a one hot vector against value idx.
    
    Parameters:
    - idx: index where you want the hot value.
    - num_class: total number of classes.
    
    Returns:
    a one hot vector in np format
    """
    return np.squeeze(np.eye(num_classes)[idx])


def generate_file_paths(
    data_dir: Path, class_name: str, video: str, window: int
) -> list[str]:
    """
    creates path fpr all the images available in the video folder.
    
    Parameters:
    - data_dir: path to parent dir with all the data.
    - class_name: class under consideration.
    - video: name of the dir named after the source video.
    - window: range of frames be selected for a single sample.
    
    Returns:
    list of paths.
    """
    video_path = os.path.join(data_dir, class_name, video)
    return [os.path.join(video_path, f"{k}.jpg") for k in range(window)]


def get_dataframe(classes: list, data_dir: str, window: int=8) -> pd.DataFrame:
    """
    created data frame from the available frame on drive.
    
    Parameters:
    - classes: List of classes in consideration.
    - data_dir: source dir of data.
    
    Returns:
    pandas dataFrame.
    """

    data = []
    for i, class_name in enumerate(tqdm(classes)):
        videos = os.listdir(os.path.join(data_dir, class_name))

        for video in videos:
            video_path = os.path.join(data_dir, class_name, video)

            for j in range(len(os.listdir(video_path)) - window):
                sliding_window = generate_file_paths(
                    data_dir, class_name, video, range(j, window + j)
                )
                data.append([sliding_window, one_hot_vector(i, len(classes))])

    df = pd.DataFrame(data)
    df.set_axis(["img", "label"], axis=1, inplace=True)
    return df


if __name__ == "__main__":
    classes = ["everythingelse", "service"]
    df = get_dataframe(classes, "/data", window=8)
    print(df.shape)
    print(df.head())
    print(len(df.img[0]))
    print(df.img[0])

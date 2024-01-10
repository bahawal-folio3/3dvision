import cv2
import numpy as np
import pandas as pd
import tensorflow as tf

from create_csv import get_dataframe


class DataGenerator(tf.keras.utils.Sequence):
    def __init__(
        self,
        df: pd.Data,
        batch_size: int = 32,
        dim: int = 172,
        num_classes: int = None,
        shuffle: bool = True,
    ):
        self.batch_size = batch_size
        self.df = df
        self.indices = self.df.index.tolist()
        self.num_classes = num_classes
        self.shuffle = shuffle
        self.on_epoch_end()
        self.window = len(df.img[0])
        self.dim = dim
        self.n_channels = 3

    def __len__(self):
        return len(self.indices) // self.batch_size

    def __getitem__(self, index):
        index = self.index[
            index * self.batch_size : (index + 1) * self.batch_size
        ]
        batch = [self.indices[k] for k in index]

        X, y = self.__get_data(batch)
        return X, y

    def on_epoch_end(self):
        self.index = np.arange(len(self.indices))
        if self.shuffle == True:
            np.random.shuffle(self.index)

    def load_images(self, slice):
        images = []
        for i in self.df.iloc[slice].img:
            image = cv2.imread(i)
            image = cv2.resize(image, (112, 112)) / 255.0
            images.append(image)
        return images, self.df.iloc[slice].label

    def __get_data(self, batch):
        """
        loads images from disk 
        
        Parameters:
        - batch: contains a batch of datapoints. Each data point has a img and it's label

        Returns:
        a batch of imgs and it's labels
        """
        X = []
        y = []
        for i, _ in enumerate(batch):
            img, label = self.load_images(batch[i])
            X.append(img)
            y.append(label)
        return np.array(X), np.array(y)


def main():
    classes = ["service", "everythingelse"]
    df = get_dataframe(classes, "data", window=8)
    input_size = 172
    dataloader = DataGenerator(df, dim=input_size)
    for i, k in dataloader:
        # test if dataloader is working corrently
        assert (
            i[0].shape[0] != input_size
        ), f"Size mismatch {input_size} != {i[0].shape[0]}"


if __name__ == "__main__":
    main()

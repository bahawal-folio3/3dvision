import tensorflow as tf
import numpy as np
import cv2

from create_csv import get_dataframe
class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, df, batch_size=32,dim=172, num_classes=None, shuffle=True):
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
        index = self.index[index * self.batch_size:(index + 1) * self.batch_size]
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

            image = cv2.resize(image,(112,112))/255.0
            images.append(image)
        # for image in images:
        #     cv2.imshow('img',image) #turn on to see frame being loaded
        #     cv2.waitKey(0) 
        return images,self.df.iloc[slice].label

    def __get_data(self, batch):
        X = []
        y = []
        for i, id in enumerate(batch):
            img, label = self.load_images(batch[i])
       
            X.append(img)
            y.append(label)
        return np.array(X),np.array(y)

if __name__=="__main__":
    classes = ['service', 'everythingelse']
    df = get_dataframe(classes, 'data', window=8)
    dataloader = DataGenerator(df, dim=172)
    for i, k in dataloader:
        for index, j in enumerate(i[0]):
            print(j.shape)
            print(k[index])
            cv2.imshow('check', j)
            cv2.waitKey(0)

        break
    
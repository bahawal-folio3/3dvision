from movinet import get_model
from dataloader import DataGenerator
from create_csv import get_dataframe
from sklearn.model_selection import train_test_split
import pandas as pd
import tensorflow as tf
import datetime

classes = ['everythingelse','service']
df = get_dataframe(classes,'data',window=8)

train_loader = DataGenerator(df)

model = get_model()


model.compile(
  optimizer=tf.keras.optimizers.Adam(),
  loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
  metrics=['acc'])

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=log_dir,
    histogram_freq=1) # Enable histogram computation for every epoch.

history = model.fit(train_loader,
                    epochs=2,
                    verbose=1,
                )
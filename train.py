from movinet import get_model
from dataloader import DataGenerator
from create_csv import get_dataframe
# from sklearn.model_selection import train_test_split
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import pandas as pd
import tensorflow as tf
import datetime

classes = ['everythingelse','service']
df = get_dataframe(classes,'data',window=5)
df = df.sample(frac=1).reset_index(drop=True)
train_size = int(df.shape[0]*0.8)
train_data = df[:train_size]
test_data = df[train_size:].reset_index(drop=True)
val_data = df[train_size:train_size+150].reset_index(drop=True)


train_loader = DataGenerator(train_data,dim=172)
test_loader = DataGenerator(test_data,dim=172)
val_loader = DataGenerator(val_data,dim=172)


model = get_model()


model.compile(
  optimizer=tf.keras.optimizers.Adam(),
  loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
  metrics=['acc', tf.keras.metrics.AUC()])

log_dir = "/logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=log_dir,
    histogram_freq=1) # Enable histogram computation for every epoch.

EPOCHS = 100
checkpoint_filepath = 'checkpoint.h5'

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=False,
    monitor='val_acc',
    mode='max',
    save_best_only=True)


history = model.fit(train_loader,
                    epochs=EPOCHS,
                    verbose=1,
                    shuffle=True,
                    validation_data=val_loader,
                    callbacks=[model_checkpoint_callback,tensorboard_callback]
                )
model.evaluate(test_loader)



from movinet import get_model, TRAINABLE
from dataloader import DataGenerator
from create_csv import get_dataframe
from f1_metrics import f1_m
# from sklearn.model_selection import train_test_split
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import pandas as pd
import tensorflow as tf
import datetime


batch_size = 24
dim = 172
window = 8
model_name = 'a2'
classes = ['service', 'hit','everythingelse']
df = get_dataframe(classes, 'data', window=window)
df = df.sample(frac=1).reset_index(drop=True)
train_size = int(df.shape[0]*0.7)
val_size = (len(df) - train_size)//2
train_data = df[:train_size]
print(len(df))
print(f"train splie: 0-{train_size-1}")
print(f"val split: {train_size}:{train_size + val_size-1}")
print(f"test split: {train_size + val_size}:{df.shape[0]}")
test_data = df[train_size+val_size:].reset_index(drop=True)
val_data = df[train_size:train_size+val_size].reset_index(drop=True)


train_loader = DataGenerator(train_data, batch_size=batch_size, dim=dim)
test_loader = DataGenerator(test_data, batch_size=batch_size, dim=dim)
val_loader = DataGenerator(val_data, batch_size=batch_size, dim=dim)


model = get_model(model_name=model_name)
# model.load_weights('f1-8-train-False.h5')

model.compile(
  optimizer=tf.keras.optimizers.Adam(),
  loss=tf.keras.losses.CategoricalCrossentropy(),
  metrics=['acc', tf.keras.metrics.AUC(),f1_m])

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(

    log_dir=log_dir,
    histogram_freq=0,
    write_graph=True,
    write_images=False,
    write_steps_per_second=False,
    update_freq='epoch',
    profile_batch=0,
    embeddings_freq=0,
    embeddings_metadata=None,
) # Enable histogram computation for every epoch.

EPOCHS = 20
checkpoint_filepath = f'{model_name}-f1-{window}-trainable{TRAINABLE}.h5'

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=False,
    monitor='f1_m',
    mode='max',
    save_best_only=True)


history = model.fit(train_loader,
                    epochs=EPOCHS,
                    verbose=1,
                    shuffle=True,
                    validation_data=val_loader,
                    callbacks=[model_checkpoint_callback, tensorboard_callback]
                )
model.evaluate(test_loader)
model.save('final-'+checkpoint_filepath)



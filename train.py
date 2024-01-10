from create_csv import get_dataframe
from dataloader import DataGenerator
from f1_metrics import f1_m
from movinet import get_model
from pathlib import Path

import argparse
import datetime
import tensorflow as tf

LOG_DIR = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

def train(
    batch_size: int,
    dim: int,
    window: int,
    model_name: str,
    train_split: float,
    classes: list,
    epochs: int,
    data_source: Path,
):
    """
    Train a movinet data on a provided data.
    
    Parameters:
    - batch_size: 
    - dim: 
    - window: 
    - model_name: 
    - train_split: 
    - classes: 
    - epochs: 
    - data_source: 
    
    Returns:
    None save a trained model
    """

    batch_size = batch_size
    dim = dim
    window = window
    model_name = window
    classes = classes
    EPOCHS = epochs

    # fetch a dataframe with all the samples
    df = get_dataframe(classes, data_source, window=window)
    # calculate the train test split using the size of df
    train_size = int(df.shape[0] * train_split)
    # break rest of the data into test and validation
    val_size = (len(df) - train_size) // 2

    train_data = df[:train_size]
    # reset the index since df maintain their original indexing
    test_data = df[train_size + val_size :].reset_index(drop=True)
    val_data = df[train_size : train_size + val_size].reset_index(drop=True)

    # use the tf sequence modules to create data generator objs
    train_loader = DataGenerator(train_data, batch_size=batch_size, dim=dim)
    test_loader = DataGenerator(test_data, batch_size=batch_size, dim=dim)
    val_loader = DataGenerator(val_data, batch_size=batch_size, dim=dim)

    model = get_model(model_name=model_name)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=["acc", tf.keras.metrics.AUC(), f1_m],
    )

    log_dir = LOG_DIR

    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir,
        histogram_freq=0,
        write_graph=True,
        write_images=False,
        write_steps_per_second=False,
        update_freq="epoch",
        profile_batch=0,
        embeddings_freq=0,
        embeddings_metadata=None,
    )  # Enable histogram computation for every epoch.

    checkpoint_filepath = f"{model_name}-{window}.h5"

    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=False,
        monitor="f1_m",
        mode="max",
        save_best_only=True,
    )

    model.fit(
        train_loader,
        epochs=EPOCHS,
        verbose=1,
        shuffle=True,
        validation_data=val_loader,
        callbacks=[model_checkpoint_callback, tensorboard_callback],
    )
    model.evaluate(test_loader)
    model.save("final-" + checkpoint_filepath)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train movinet")
    parser.add_argument(
        "--batch_size",
        required=True,
        help="Path to the source weights file",
        type=str,
    )
    parser.add_argument(
        "--dim", required=True, help="Path to the source video file", type=str,
    )
    parser.add_argument(
        "--window",
        required=True,
        help="Path to the target video file (output)",
        type=str,
    )
    parser.add_argument(
        "--model_name",
        default=None,
        help="Version of movinet to be used",
        type=str,
    )
    parser.add_argument(
        "--classes",
        default=["hit", "serve", "everything_else"],
        help="list of classes",
        type=list[str],
    )
    parser.add_argument(
        "--train_split",
        default=0.8,
        help="split size for training",
        type=float,
    )
    parser.add_argument(
        "--epochs",
        default=100,
        help="number of iteration we have to train this model",
        type=float,
    )
    parser.add_argument(
        "--data_source",
        default="data/",
        help="path to the data source diractory",
        type=ste,
    )
    args = parser.parse_args()

    train(
        batch_size=args.batch_size,
        dim=args.dim,
        window=args.window,
        model_name=args.model_name,
        train_split=args.train_split,
        classes=args.classes,
        epochs=args.epochs,
        data_source=args.data_source,
    )

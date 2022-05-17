import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
import tensorflow_hub as hub

def get_model():
    hub_url = "https://tfhub.dev/tensorflow/movinet/a1/base/kinetics-600/classification/3"

    encoder = hub.KerasLayer(hub_url, trainable=True)

    inputs = tf.keras.layers.Input(
        shape=[None, None, None, 3],
        dtype=tf.float32,
        name='image')

    # [batch_size, 600]
    outputs = encoder(dict(image=inputs))

    # example_input = tf.ones([1, 8, 172, 172, 3])
    # example_output = model(example_input)


    model = Sequential([

        tf.keras.Model(inputs, outputs, name='movinet'),
        
        Dense(4096, activation='relu', name='fc6'),
        Dropout(.5),
        Dense(4096, activation='relu', name='fc7'),
        Dropout(.5),
        Dense(1, activation='sigmoid', name='fc8'),
    ])
    return model

if __name__ == "__main__":
    model = get_model()
    example_input = tf.ones([32, 8, 112, 112, 3])
    example_output = model(example_input)
    print(example_output)

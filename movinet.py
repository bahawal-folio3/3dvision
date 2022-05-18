import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten,BatchNormalization
import tensorflow_hub as hub
TRAINABLE = False
model_urls = {
    'a0':'https://tfhub.dev/tensorflow/movinet/a0/base/kinetics-600/classification/3',
    'a1':'https://tfhub.dev/tensorflow/movinet/a1/base/kinetics-600/classification/3',
    'a2':'https://tfhub.dev/tensorflow/movinet/a3/base/kinetics-600/classification/3',
    'a3':'https://tfhub.dev/tensorflow/movinet/a4/base/kinetics-600/classification/3',
    'a4':'https://tfhub.dev/tensorflow/movinet/a4/base/kinetics-600/classification/3',

}
def get_model(model_name = 'a1'):

    hub_url = model_urls.get(model_name,"'https://tfhub.dev/tensorflow/movinet/a0/steam/kinetics-600/classification/3'")

    encoder = hub.KerasLayer(hub_url, trainable=TRAINABLE)

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
        BatchNormalization(),
        Dense(4096, activation='relu', name='fc6'),
        Dropout(0.5),
        Dense(2, activation='softmax', name='fc9'),
    ])
    return model

if __name__ == "__main__":
    model = get_model()
    example_input = tf.ones([32, 8, 112, 112, 3])
    example_output = model(example_input)
    print(example_output)

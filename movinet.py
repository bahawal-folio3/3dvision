import tensorflow as tf
import tensorflow_hub as hub

from tensorflow.keras.layers import BatchNormalization, Dense, Dropout, Flatten
from tensorflow.keras.models import Sequential


TRAINABLE = False

model_urls = {
    "a0": "https://tfhub.dev/tensorflow/movinet/a0/base/kinetics-600/classification/3",
    "a1": "https://tfhub.dev/tensorflow/movinet/a1/base/kinetics-600/classification/3",
    "a2": "https://tfhub.dev/tensorflow/movinet/a3/base/kinetics-600/classification/3",
    "a3": "https://tfhub.dev/tensorflow/movinet/a4/base/kinetics-600/classification/3",
    "a4": "https://tfhub.dev/tensorflow/movinet/a4/base/kinetics-600/classification/3",
}


def get_model(model_name: str = "a1") -> Sequential:
    """
            create a model with movinet as backbone
            
            Parameters:
            - model_name: version of movinet to be used.            
            Returns:
            tensorflow Sequential as model.
            """
    hub_url = model_urls.get(
        model_name,
        "https://tfhub.dev/tensorflow/movinet/a0/steam/kinetics-600/classification/3",
    )

    encoder = hub.KerasLayer(hub_url, trainable=TRAINABLE)
    # None make the input shape dynamic shape = [b,h,w,c]
    inputs = tf.keras.layers.Input(
        shape=[None, None, None, 3], dtype=tf.float32, name="image"
    )
    outputs = encoder(dict(image=inputs))

    model = Sequential(
        [
            tf.keras.Model(inputs, outputs, name="movinet"),
            BatchNormalization(),
            Dense(4096, activation="relu", name="fc6"),
            Dropout(0.5),
            Dense(2, activation="softmax", name="fc9"),
        ]
    )
    return model


if __name__ == "__main__":
    model = get_model()
    example_input = tf.ones([32, 8, 112, 112, 3])
    example_output = model(example_input)
    print(example_output)

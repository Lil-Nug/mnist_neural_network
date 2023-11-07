import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

data = keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = data.load_data()

x_train = x_train / 255
x_test = x_test / 255

model = keras.Sequential(
    [
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dense(10, activation="softmax"),
    ]
)

model.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)

model.fit(x_train, y_train, epochs=5)

prediction = model.predict(x_test)

for i in range(10):
    print(f"image {i}")
    print("actual", y_test[i])
    print("prediction", np.argmax(prediction[i]))

import os 
import tensorflow as tf
import keras
from keras import layers

mnist = keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train, x_test = x_train / 255.0, x_test / 255.0
# 28, 28
# input_size=28
# sequence_length = 28

model = keras.models.Sequential()
model.add(keras.Input(shape=(28, 28))) # sequence_length, input_size
# model.add(layers.SimpleRNN(128, return_sequences=True, activation='relu'))  # N, 28, 128
model.add(layers.SimpleRNN(128, return_sequences=False, activation='relu'))  # N, 128

model.add(layers.Dense(10))

print(model.summary())

loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optim = keras.optimizers.Adam(learning_rate=0.001)
metrics = ["accuracy"]

model.compile(loss=loss, optimizer=optim, metrics=metrics)

batch_size = 64
epochs = 5

model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=2)

model.evaluate(x_test, y_test, batch_size=batch_size, verbose=2)
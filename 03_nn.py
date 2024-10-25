import os 
import numpy as np
import tensorflow as tf
from tensorflow import keras 
import matplotlib.pyplot as plt 
from keras.models import Sequential

mnist = keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

x_train, x_test = x_train / 255.0, x_test / 255.0

# model
model = Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10)
])

print(model.summary())

# another way 
# model = Sequential()
# model.add(keras.layers.Flatten(input_shape=(28, 28)))
# model.add(keras.layers.Dense(128, activation='relu'))
# model.add(keras.layers.Dense(10))

loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optim = keras.optimizers.Adam(learning_rate=0.001)
metrics = ["accuracy"]

model.compile(loss=loss, optimizer=optim, metrics=metrics)

batch_size = 64
epochs = 5

model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, shuffle=True, verbose=2)

# evaluate
model.evaluate(x_test, y_test, batch_size=batch_size, verbose=2)

# predictions

# 1. option: build new model with softmax layer
probability_model = keras.models.Sequential([
    model,
    keras.layers.Softmax()
])

predictions = probability_model(x_test)
pred0 = predictions[0]
print(pred0)

label0 = np.argmax(pred0)
print(label0)

# 2. Option: original model + nn.softmax, call model(x)
predictions = model(x_test)
predictions = tf.nn.softmax(predictions)
pred0 = predictions[0]
print(pred0)
label0 = np.argmax(pred0)
print(label0)

# 3. option: original model + nn.softmax, call model.predict(x)
predictions = model.predict(x_test, batch_size=batch_size)
predictions = tf.nn.softmax(predictions)
pred0 = predictions[0]
print(pred0)
label0 = np.argmax(pred0)
print(label0)

# call argmax for multiple labels
pred05s = predictions[0:5]
print(pred05s.shape)
label05s = np.argmax(pred05s, axis=1)
print(label05s)
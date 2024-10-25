import os 

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers 

import matplotlib.pyplot as plt

cifar10 = keras.datasets.cifar10

(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

print(train_images.shape)  # (50000, 32, 32, 3)

train_images, test_images = train_images / 255.0 , test_images / 255.0

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog',
               'frog', 'horse', 'ship', 'truck']

def show():
    plt.figure(figsize=(10, 10))
    for i in range(16):
        plt.subplot(4, 4, i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(train_images[i], cmap=plt.cm.binary)
        plt.xlabel(class_names[train_labels[i][0]])

show()


model = keras.models.Sequential()
model.add(layers.Conv2D(32, (3, 3), strides=(1, 1), padding='valid', activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPool2D((2,2)))
model.add(layers.Conv2D(32, 3, strides=(1, 1), activation='relu'))
model.add(layers.MaxPool2D((2,2)))

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
# we added logits = True to our loss, so no activation required at end
model.add(layers.Dense(10))

print(model.summary())
# import sys; sys.exit()

loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optim = keras.optimizers.Adam(learning_rate=0.001)
metrics = ["accuracy"]

model.compile(optimizer=optim, loss=loss, metrics=metrics)

batch_size = 64
epochs = 5

model.fit(train_images, train_labels, epochs=epochs, batch_size=batch_size, verbose=2)

model.evaluate(test_images, test_labels, batch_size=batch_size, verbose=2)

# 1) Save whole model
# SavedModel, HDF5

model.save('nn.h5')
# model.save("neural_net.keras")

new_model = keras.models.load_model("nn.h5")
new_model.evaluate(test_images, test_labels, verbose=2)


# 2) Save only weights
model.save_weights("nn_weights.weights.h5")

# initiate
model.load_weights("nn_weights.weights.h5")


# 3) Save only architecture, to_json
json_string = model.to_json()

with open("nn_model", "w") as f:
    f.write(json_string)

with open("nn_model", "r") as f:
    loaded_json_string = f.read()


new_model = keras.models.model_from_json(loaded_json_string)
print(new_model)
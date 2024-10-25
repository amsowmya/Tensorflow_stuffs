import tensorflow as tf
from tensorflow import keras 
from tensorflow.keras import layers 


model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10)
])

print(model.summary())


# functional API
inputs = keras.Input(shape=(28, 28))

flatten = keras.layers.Flatten()
dense1 = keras.layers.Dense(128, activation='relu')
dense2 = keras.layers.Dense(10)

x = flatten(inputs)
x = dense1(x)
outputs = dense2(x)

model = keras.Model(inputs = inputs, outputs = outputs, name='functional_model')

print(model.summary())

################## MULTIPLE OUTPUTS ##################
inputs = keras.Input(shape=(28, 28))

flatten = keras.layers.Flatten()
dense1 = keras.layers.Dense(128, activation='relu')
dense2 = keras.layers.Dense(10)
dense2_2 = keras.layers.Dense(1)

x = flatten(inputs)
x = dense1(x)
outputs = dense2(x)
outputs2 = dense2_2(x)

model = keras.Model(inputs = inputs, outputs = [outputs, outputs2], name='functional_model')

print(model.summary())

################## MULTIPLE OUTPINPUTSUTS ##################

'''
# Convert functional api to sequential -> If it is linear
new_model = keras.models.Sequential()
for layer in model.layers:
    new_model.add(layer)

# Convert functional api to sequential -> If it is non-linear
inputs = keras.Input(shape=(28, 28))
x = new_model.layers[0](inputs)
for layer in new_model.layers[1:-1]:
    x = layer(x)
outputs = x
'''

##################### ADVANTAGES OF FUNCTIONAL API
# Models with multiple inputs and outputs
# Shared layers
# Extract and reuse nodes in the graph of layers
# (Model are callable like layers (put model into sequential))

inputs = model.inputs
outouts = model.outputs

input0 = model.layers[1].input
output0 = model.layers[1].output

print(inputs)
print(outputs)
print(input0)
print(output0)

# Transfor learning
base_model = keras.applications.VGG16()

x = base_model.layers[-2].output
new_output = keras.layers.Dense(1)(x)

new_model = keras.Model(inputs=base_model.inputs, outputs=new_output)
import numpy as np
from tensorflow import keras

Input = keras.layers.Input
Dense = keras.layers.Dense
Model = keras.models.Model
Dropout = keras.layers.Dropout
Sequential = keras.models.Sequential
Activation = keras.layers.Activation

model = Sequential([
    Dense(512, input_shape=(784,)),
    Activation("relu"),
    Dense(256),
    Activation("relu"),
    Dense(128),
    Activation("relu"),
    Dropout(0.5),
    Dense(10),
    Activation("softmax")
])
data = np.random.random((10240, 784))
label = np.random.randint(2, size=(10240, 10))

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy']
              )
model.fit(data, label, epochs=100000, batch_size=64)

'''
Specifying the input shape

The model needs to know what input shape it should expect. For this reason, the first layer in a Sequential model (and only the first, because following layers can do automatic shape inference) needs to receive information about its input shape. There are several possible ways to do this:

    Pass an input_shape argument to the first layer. This is a shape tuple (a tuple of integers or None entries, where None indicates that any positive integer may be expected). In input_shape, the batch dimension is not included.
    Some 2D layers, such as Dense, support the specification of their input shape via the argument input_dim, and some 3D temporal layers support the arguments input_dim and input_length.
    If you ever need to specify a fixed batch size for your inputs (this is useful for stateful recurrent networks), you can pass a batch_size argument to a layer. If you pass both batch_size=32 and input_shape=(6, 8) to a layer, it will then expect every batch of inputs to have the batch shape (32, 6, 8)
'''

'''
model = Sequential()
model.add(Dense(32, input_dim=784))
model.add(Activation("relu"))

# As such, the following snippets are strictly equivalent:
model = Sequential()
model.add(Dense(32, input_shape=(784,)))

model = Sequential()
model.add(Dense(32, input_dim=784))
'''

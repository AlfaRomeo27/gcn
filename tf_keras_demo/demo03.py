from tensorflow import keras

Sequential = keras.models.Sequential
Dense = keras.layers.Dense
Dropout = keras.layers.Dropout
Activation = keras.layers.Activation
SGD = keras.optimizers.SGD

# Generate numpy data
import numpy as np

x_train = np.random.random((10000, 20))
y_train = keras.utils.to_categorical(np.random.randint(10, size=(10000, 1)), num_classes=10)

x_test = np.random.random((10000, 20))
y_test = keras.utils.to_categorical(np.random.randint(10, size=(10000, 1)), num_classes=10)

model = Sequential()
model.add(Dense(64, activation='relu', input_dim=20))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5)),
model.add(Dense(10, activation='softmax'))

sgd = SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy']
              )
model.fit(x_train, y_train, batch_size=100, epochs=1000)

score = model.evaluate(x_test, y_test, batch_size=100)

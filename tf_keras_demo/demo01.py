import tensorflow as tf
from tensorflow import keras
import numpy as np

data = np.ones(shape=[10, 4, 4, 3])
labels = np.ones(shape=[10, 100])
print(data.shape)

layer = keras.layers

print(tf.VERSION)
print(tf.keras.__version__)



model = tf.keras.Sequential([

    layer.Conv2D(32, (3, 3), 1, 'SAME', input_shape=(4, 4, 3)),
    layer.BatchNormalization(),
    layer.Activation('relu'),

    layer.Conv2D(64, 3, 1, 'SAME'),
    layer.BatchNormalization(),
    layer.Activation('relu'),

    layer.MaxPool2D(2, 2),

    layer.Conv2D(128, (3, 3), 1, 'SAME'),
    layer.BatchNormalization(),
    layer.Activation('relu'),

    layer.Conv2D(256, (3, 3), 1, 'SAME'),
    layer.BatchNormalization(),
    layer.Activation('relu'),

    layer.MaxPool2D(2, 2),

    layer.Conv2D(512, (3, 3), 1, 'SAME'),
    layer.BatchNormalization(),

    layer.Reshape((-1, 512)),

    layer.Dense(256, activation='relu'),

    layer.Dense(128, activation='relu'),

    layer.Dense(100, activation="softmax")
])
model.compile(optimizer=tf.train.AdamOptimizer(0.0001),
              loss='categorical_crossentropy')

model.fit(data, labels, epochs=10, batch_size=10)

'''

###示例代码
def function_1(A):
    print("function_1")


def function_2(B):
    print(B(3))
    print("function_2")


@function_1
@function_2
def function_name(n):
    print("Hello World ,i am function_name")
    return n + 5
'''

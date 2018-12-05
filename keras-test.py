# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

x = np.array(([0,0],[0,1],[1,0],[1,1]), dtype=float)
y = np.array(([0],[0],[0],[1]), dtype=float)

def plot_history(history):
  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Abs Error [1000$]')
  plt.plot(history.epoch, np.array(history.history['mean_absolute_error']),
           label='Train Loss')
  plt.plot(history.epoch, np.array(history.history['val_mean_absolute_error']),
           label = 'Val loss')
  plt.legend()
  plt.ylim([0, 5])

model = keras.Sequential([
    keras.layers.Dense(3, activation=tf.nn.sigmoid),
    keras.layers.Dense(1, activation=tf.nn.sigmoid)
])

model.compile(optimizer=tf.train.GradientDescentOptimizer(1),
              loss='mse',
              metrics=['mae'])

history = model.fit(x, y, verbose=0, epochs=2000)

predictions = model.predict(x[0])
# plot_history(history)
# print(history.history)
print(predictions)

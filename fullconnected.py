import json

import numpy as np
from tensorflow import keras

with open('training/naive-mar2019_861k_spell.json') as f:
    data = json.load(f)

train_queries = np.array([spell['x'] for spell in data])
train_labels = np.array([spell['y'] for spell in data])

print(f"X shape: {train_queries.shape}")
print(f"Y shape: {train_labels.shape}")

model = keras.Sequential([
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(501, activation='softmax')
])

model.compile(optimizer=keras.optimizers.Adam(lr=0.002),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x=train_queries, y=train_labels, epochs=15, validation_split=0.03,
          callbacks=[
              keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, batch_size=32, write_graph=True,
                                          write_grads=True, write_images=True, update_freq='epoch'),
              keras.callbacks.EarlyStopping('val_loss', min_delta=-0.005, patience=10, verbose=1)
          ])

model.summary()

test_loss, test_acc = model.evaluate(train_queries, train_labels)
print('Test accuracy:', test_acc)

fileout = input("Save weights? (enter weight name) ")
model.save(f"models/{fileout}.h5")

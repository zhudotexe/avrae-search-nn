import json

import numpy as np
from tensorflow import keras

with open('training/1-mar2019_861k_spell.json') as f:
    data = json.load(f)

train_queries = np.array([spell['x'] for spell in data])
train_labels = np.array([spell['y'] for spell in data])

print(f"X shape: {train_queries.shape}")
print(f"Y shape: {train_labels.shape}")

model = keras.Sequential([
    keras.layers.Dense(150, activation='relu'),
    keras.layers.Dropout(0.1),
    keras.layers.Dense(501, activation='softmax')
])

model.compile(optimizer=keras.optimizers.Adam(lr=0.01),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x=train_queries, y=train_labels, epochs=1000, validation_split=0.03, shuffle=True,
          callbacks=[
              keras.callbacks.EarlyStopping('val_loss', min_delta=0, patience=20, restore_best_weights=True, verbose=1),
              keras.callbacks.ReduceLROnPlateau('val_loss', factor=0.2, patience=5, min_lr=0.001, verbose=1)
          ])

# test_loss, test_acc = model.evaluate(val_queries, val_labels)

# print('Test accuracy:', test_acc)

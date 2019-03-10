import json

import numpy as np
from tensorflow import keras

# with open('training/2-mar2019_861k_spell.json') as f:
#     data = json.load(f)
#
# train_queries = np.array([spell['x'] for spell in data])
# train_queries = np.expand_dims(train_queries, axis=2)
# train_labels = np.array([spell['y'] for spell in data])

with open('training/naive-mar2019_861k_spell.json') as f:
    data = json.load(f)

train_queries = np.array([spell['x'] for spell in data])
train_queries = np.expand_dims(train_queries, axis=2)
train_labels = np.array([spell['y'] for spell in data])

print(f"X shape: {train_queries.shape}")
print(f"Y shape: {train_labels.shape}")

model = keras.Sequential([
    keras.layers.Conv1D(25, 2, activation='relu', input_shape=(16, 1)),
    keras.layers.AveragePooling1D(),
    # keras.layers.Conv1D(15, 3, activation='relu'),
    # keras.layers.MaxPool1D(),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(501, activation='softmax')
])

model.compile(optimizer=keras.optimizers.Adam(lr=0.001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

model.fit(x=train_queries, y=train_labels, epochs=25, validation_split=0.05, shuffle=True,
          callbacks=[
              keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, batch_size=32, write_graph=True,
                                          write_grads=True, write_images=True, embeddings_freq=0,
                                          embeddings_layer_names=None, embeddings_metadata=None, embeddings_data=None,
                                          update_freq='epoch'),
              #keras.callbacks.ReduceLROnPlateau('val_loss', patience=10, verbose=1, min_lr=0.0002),
              keras.callbacks.EarlyStopping('val_loss', min_delta=0, patience=40, verbose=1)
          ])

test_loss, test_acc = model.evaluate(train_queries, train_labels)
print('Test accuracy:', test_acc)

fileout = input("Save weights? (enter weight name) ")
model.save(f"models/{fileout}.h5")

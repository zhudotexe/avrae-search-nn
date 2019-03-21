"""
Results:
magic1_embedding_conv Pure: t1=13852 t2=944 t3=335 t10=686 f=1110 t=39.88
    Embedding 29x16 -> Dropout 0.2 -> Conv1D 25, 3 -> AvgPool -> Dropout 0.2 -> Dense 501
magic1_embedding_conv Mixed: t1=13821 t2=809 t3=390 t10=1125 f=782 t=149.48

magic1_embedding_conv_maxpool Pure: t1=13945 t2=949 t3=346 t10=626 f=1061 t=39.90
    Embedding 29x16 -> Dropout 0.2 -> Conv1D 25, 3 -> MaxPool -> Dropout 0.2 -> Dense 501
magic1_embedding_conv_maxpool Mixed: t1=13901 t2=815 t3=380 t10=1095 f=736 t=154.63

magic1_embedding_conv_smaller Pure: t1=12979 t2=1172 t3=470 t10=976 f=1330 t=39.28
    Embedding 29x16 -> Dropout 0.2 -> Conv1D 75, 3 -> GlobAvgPool -> Dropout 0.2 -> Dense 501
magic1_embedding_conv_smaller Mixed: t1=12814 t2=961 t3=495 t10=1546 f=1111 t=163.57
"""

import json
import sys

import numpy as np
from tensorflow import keras

SRD = 'srd' in sys.argv

with open(f'training/embedding-{"srd-" if SRD else ""}mar2019_861k_spell.json') as f:
    data = json.load(f)

train_queries = np.array([spell['x'] for spell in data])  # 16d list of integers
train_labels = np.array([spell['y'] for spell in data])  # 501d vector

print(f"X shape: {train_queries.shape}")
print(f"Y shape: {train_labels.shape}")

model = keras.Sequential([
    keras.layers.Embedding(29, 16, input_length=16),
    keras.layers.SpatialDropout1D(0.2),
    keras.layers.Conv1D(25, 3, activation='relu', padding='same'),
    keras.layers.MaxPool1D(),
    keras.layers.Flatten(),
    # keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(298 if SRD else 501, activation='softmax')
])

model.compile(optimizer=keras.optimizers.Adam(lr=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

model.fit(x=train_queries, y=train_labels, epochs=1000, validation_split=0.05, shuffle=True,
          callbacks=[
              keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, batch_size=32, write_graph=True,
                                          write_grads=True, write_images=True, embeddings_freq=0,
                                          embeddings_layer_names=None, embeddings_metadata=None, embeddings_data=None,
                                          update_freq='epoch'),
              # keras.callbacks.ReduceLROnPlateau('val_loss', patience=10, verbose=1, min_lr=0.0002),
              keras.callbacks.EarlyStopping('val_loss', min_delta=0, patience=40, verbose=1)
          ])

test_loss, test_acc = model.evaluate(train_queries, train_labels)
print('Test accuracy:', test_acc)

fileout = input("Save weights? (enter weight name) ")
model.save(f"models/{fileout}.h5")

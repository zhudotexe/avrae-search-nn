import json

import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt


def load_spell_data():
    with open('../training/spell.json') as f:
        data = json.load(f)

    train = np.array([spell['tokenized'] for spell in data])
    labels = np.array([spell['result'] for spell in data])

    # with open('../preprocessing/map-spell.json') as f:
    #     labels = json.load(f)
    #     labels = list(labels.values())

    return train, labels


train_queries, train_labels = load_spell_data()

test_queries = []
test_labels = []

model = keras.Sequential([
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(497, activation=tf.nn.softmax)
])

model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

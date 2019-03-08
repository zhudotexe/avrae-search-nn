import json

import numpy as np
import tensorflow as tf

from preprocess import MAGIC_2, clean, tokenize

modelname = input("Model: ")
model = tf.keras.models.load_model(f'models/{modelname}.h5')

with open('preprocessing/map-mar2019_861k_spell.json') as f:
    map_ = json.load(f)


def get_predictions(query):
    query = clean(query)
    query = tokenize(query, MAGIC_2)
    query = np.expand_dims(query, 0)
    if 'conv' in modelname:
        query = np.expand_dims(query, 2)

    prediction = model.predict(query)
    prediction = prediction[0]

    indexed = list(enumerate(prediction))
    weighted = sorted(indexed, key=lambda e: e[1], reverse=True)

    print('\n'.join([f"{map_[str(r[0])]}: {r[1]:.2f}" for r in weighted[:10]]))
    print()


if __name__ == '__main__':
    while True:
        query = input("Query: ").strip()
        get_predictions(query)

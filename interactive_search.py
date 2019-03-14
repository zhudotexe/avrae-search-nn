import json

import numpy as np
import tensorflow as tf

from preprocess import clean, tokenize, MAGIC_1, MAGIC_2

modelname = input("Model: ")
model = tf.keras.models.load_model(f'models/{modelname}.h5')

model.summary()

with open('preprocessing/map-mar2019_861k_spell.json') as f:
    map_ = json.load(f)


def get_predictions(query, model_name, magic_string):
    query = clean(query)
    query = tokenize(query, magic_string, 'embedding' in model_name)
    query = np.expand_dims(query, 0)
    if 'conv' in model_name and 'embedding' not in model_name:
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
        get_predictions(query, modelname, MAGIC_1 if modelname.startswith('magic1') else MAGIC_2)

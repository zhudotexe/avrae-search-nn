import json

from preprocess import MAGIC_2, clean, tokenize

if __name__ == '__main__':
    with open('training/2-mar2019_861k_spell.json') as f:
        data = json.load(f)
    with open('preprocessing/map-mar2019_861k_spell.json') as f:
        map_ = json.load(f)

    while True:
        token_to_find = tokenize(clean(input("Query? ")), MAGIC_2)
        entry = next((d for d in data if d['x'] == token_to_find), None)
        if not entry:
            print("Token not found")
            continue
        print(f"X: {token_to_find}")
        print("Y:")
        print(entry['y'])
        for y, w in enumerate(entry['y']):
            if w:
                print(f"{map_.get(str(y))}: {w}")
        print()

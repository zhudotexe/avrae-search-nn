"""
In: an unprocessed sorted query file from ../training/unprocessed/
Out: a map file mapping output indices to spell names (for use in dataparser)
     Tokenized query files (in ../training/)
"""
import json
import time

from mapper import map_data

MAGIC_1 = "abcdefghijklmnopqrstuvwxyz '"
MAGIC_2 = "qwertyuiopasdfghjkl'zxcvbnm "
INPUT_LENGTH = 16


def clean_inputs(filename):
    with open(f"../training/{filename}") as f:
        data = json.load(f)

    print("Cleaning queries...")
    for entry in data:
        entry['query'] = clean_query(entry['query'])

    with open(f"../training/old_{filename}", 'w') as f:
        json.dump(data, f, indent=2)
    print("Done cleaning.")


def clean_query(query):
    filtered = query.lower()
    filtered = ''.join(c for c in filtered if c in MAGIC_1)
    return filtered[:INPUT_LENGTH].strip()


def tokenize(path):
    with open(path) as f:
        data = json.load(f)

    print("Tokenizing queries...")
    num_chars = len(MAGIC_1)
    for entry in data:
        tokenized = [0.] * INPUT_LENGTH
        tokenized2 = [0.] * INPUT_LENGTH
        for i, char in enumerate(entry['query']):
            tokenized[i] = (MAGIC_1.index(char) + 1) / num_chars
            tokenized2[i] = (MAGIC_2.index(char) + 1) / num_chars
        entry['tokenized_1'] = tokenized
        entry['tokenized_2'] = tokenized2

    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
    print("Done tokenizing.")


if __name__ == '__main__':
    filename = input("Filename: ").strip()
    starttime = time.time()
    map_, reverse_map = map_data(filename)
    clean_inputs(filename)
    tokenize(f"../training/old_{filename}")
    endtime = time.time()
    print(f"Done! Took {endtime-starttime:.3f} seconds.")

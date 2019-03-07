"""
In: an unprocessed sorted query file from ../training/unprocessed/
Out: a map file mapping output indices to spell names (for use in dataparser)
"""

import json
import time


def map_data(filename):
    typename = filename.split('_')[-1]
    with open(f"res/{typename}") as f:
        data = json.load(f)

    # generate map
    print("Generating map...")
    mapped = {}
    reverse_map = {}
    for i, entry in enumerate(data):
        mapped[i] = entry['name']
        reverse_map[entry['name']] = i

    # dump map
    print("Dumping map...")
    with open(f"preprocessing/map-{filename}", 'w') as f:
        json.dump(mapped, f, indent=2)

    # map training
    print("Mapping training...")
    out = []
    with open(f"training/unprocessed/{filename}") as f:
        training = json.load(f)

    for entry in training:
        entry['result'] = reverse_map[entry['result']]
        out.append(entry)

    with open(f"training/{filename}", 'w') as f:
        json.dump(out, f, indent=2)
    print("Done mapping.")
    return mapped, reverse_map


if __name__ == '__main__':
    filename = input("Filename: ").strip()
    starttime = time.time()
    map_, reverse_map = map_data(filename)
    endtime = time.time()
    print(f"Done! Took {endtime-starttime:.3f} seconds.")

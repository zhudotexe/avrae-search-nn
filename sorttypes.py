"""
Sorts a data dump file into query types.
Input: Raw query file (root dir)
Outputs: 1 query file per type, in training/unprocessed/[BATCH]_[TYPE].json
"""
import json

infile = input("Infile? ")

with open(infile) as f:
    data = json.load(f)

types = {}

for entry in data:
    print(f"{entry['type']}: {entry['query']} => {entry['result']}")
    if entry['type'] not in types:
        types[entry['type']] = []
    types[entry['type']].append({
        'query': entry['query'],
        'result': entry['result']
    })

for type_, entries in types.items():
    with open(f'training/unprocessed/{infile[:-5]}_{type_}.json', 'w') as f:
        json.dump(entries, f, indent=2)

print(f"Sorted {len(data)} entries into {len(types)} categories")

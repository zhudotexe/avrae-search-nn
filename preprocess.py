"""
Sorts a raw type query file into mapped training data (ready to plug in to keras)
Input: Raw type query file (in training/unprocessed/[BATCH]_[TYPE].json)
       Result objects file (in res/[TYPE].json)
Outputs: mapped training file, in training/[BATCH]_[TYPE].json
         map file, in preprocessing/map-[BATCH]_[TYPE].json
"""
import collections
import json
import time

MAGIC_1 = "abcdefghijklmnopqrstuvwxyz '"
MAGIC_2 = "qwertyuiopasdfghjkl'zxcvbnm "
# MAGIC_2 = "aqzswxdecfrvgtb hynjumkilop'"
INPUT_LENGTH = 16


def load_type_query_file(name):
    """
    Loads a list of unprocessed type queries from a file in training/unprocessed.
    {
        "query": string,
        "result": string
    }
    """
    with open(f"training/unprocessed/{name}") as f:
        data = json.load(f)
    print(f"Loaded {len(data)} queries from {name}")
    return data


def map_data(queries, filename):
    """
    Generates a map (i -> name) and reverse map (name -> i), and modifies queries to set result to an int.
    Output:
    {
        "query": string,
        "result": int
    }
    """
    typename = filename.split('_')[-1]
    with open(f"res/{typename}") as f:
        result_objs = json.load(f)

    # generate map
    print("Generating map...")
    mapped = {}
    reverse_map = {}
    for i, entry in enumerate(result_objs):
        mapped[i] = entry['name']
        reverse_map[entry['name']] = i

    # dump map
    print("Dumping map...")
    with open(f"preprocessing/map-{filename}", 'w') as f:
        json.dump(mapped, f, indent=2)

    # map training
    print("Mapping queries...")

    for entry in queries:
        entry['result'] = reverse_map[entry['result']]

    print("Done mapping.")
    return mapped, reverse_map


def clean_queries(data):
    print("Cleaning queries...")
    for entry in data:
        entry['query'] = clean(entry['query'])
    print("Done.")


def clean(query):
    filtered = query.lower()
    filtered = ''.join(c for c in filtered if c in MAGIC_1)
    return filtered[:INPUT_LENGTH].strip()


def clean_dupes(data):
    """
    Cleans up a long list of queries into what each query returned.
    output:
    {
        "[query]": {counter of results (int: int)}
    }
    """
    print("Cleaning up duplicates...")
    queries = collections.defaultdict(lambda: collections.Counter())
    for entry in data:
        query = entry['query']
        result = entry['result']
        queries[query][result] += 1
    print(f"Cleaned {len(data)} entries into {len(queries)}.")
    return queries


def dump_training(cleaned, filename, num_results):
    print("Formatting for training...")
    out1 = []
    out2 = []
    for query, results in cleaned.items():
        tokenized = tokenize(query, MAGIC_1)
        tokenized2 = tokenize(query, MAGIC_2)
        result_vec = generate_y_vector(results, num_results)
        out1.append({'x': tokenized, 'y': result_vec})
        out2.append({'x': tokenized2, 'y': result_vec})
    with open(f'training/1-{filename}', 'w') as f:
        json.dump(out1, f)
    with open(f'training/2-{filename}', 'w') as f:
        json.dump(out2, f)
    print("Done formatting.")


def dump_training_2(cleaned, filename, num_results):
    print("Formatting for naive training...")
    out2 = []
    for query, results in cleaned.items():
        tokenized2 = tokenize(query, MAGIC_2)
        for i, count in results.items():
            # for _ in range(count):
            vec = [0.] * num_results
            vec[i] = 1
            out2.append({'x': tokenized2, 'y': vec})
    with open(f'training/naive-{filename}', 'w') as f:
        json.dump(out2, f)
    print("Done formatting.")


def tokenize(query, magic_string):
    num_chars = len(magic_string)
    tokenized = [0.] * INPUT_LENGTH
    for i, char in enumerate(query):
        tokenized[i] = (magic_string.index(char) + 1) / num_chars
    return tokenized


def generate_y_vector(results, num_results):
    """Given a count of results and the total number of results, returns a normalized label vector."""
    vec = [0.] * num_results
    for i, count in results.items():
        vec[i] = count
    vec_sum = sum(vec)
    for i, _ in enumerate(vec):
        vec[i] /= vec_sum
    return vec


if __name__ == '__main__':
    filename = input("Filename: ").strip()
    starttime = time.time()
    data = load_type_query_file(filename)
    map_, reverse_map = map_data(data, filename)
    clean_queries(data)
    cleaned = clean_dupes(data)
    dump_training(cleaned, filename, len(map_))
    dump_training_2(cleaned, filename, len(map_))

    endtime = time.time()
    print(f"Done! Took {endtime-starttime:.3f} seconds.")

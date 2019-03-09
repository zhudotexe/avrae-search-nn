import collections
import json

import matplotlib.pyplot as plt
import numpy as np

infile = input("Infile? ")

searches = collections.Counter()
results = collections.Counter()

with open(infile) as f:
    data = json.load(f)

i = 0
for entry in data:
    query = entry['query']
    result = entry['result']

    searches[query] += 1
    results[result] += 1
    i += 1
    if i % 1000 == 0:
        print(i)

search_vals = np.array(sorted(list(searches.values()), reverse=True))
result_vals = np.array(sorted(list(results.values()), reverse=True))

to_graph = result_vals

plt.bar(list(range(len(result_vals))), to_graph)

plt.xlabel('Result Index')
plt.ylabel('Count')
plt.title('Most Common Results')
plt.axis([0, len(result_vals), 0, max(result_vals) + 50])
plt.grid(True)
plt.show()

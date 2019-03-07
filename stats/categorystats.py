import collections
import json

infile = input("Infile? ")

searches = collections.Counter()
results = collections.Counter()
vagueness = collections.defaultdict(lambda: set())

with open(infile) as f:
    data = json.load(f)

i = 0
for entry in data:
    query = entry['query']
    result = entry['result']

    searches[query] += 1
    results[result] += 1
    vagueness[query].add(result)
    i += 1
    if i % 1000 == 0:
        print(i)

print(f"Processed {i} searches")

lower_results = set(r.lower() for r in results.keys())
nfm_searches = collections.Counter()
for query in searches.keys():
    if query.lower() not in lower_results:
        nfm_searches[query.lower()] += 1

vague = list(sorted(((k, len(v)) for k, v in vagueness.items()), key=lambda p: p[1], reverse=True))

long = list(sorted((e['query'] for e in data), key=lambda q: len(q), reverse=True))

outfile = infile.split('/')[-1][:-5]

rpt = f"{outfile}.json\n" \
      f"{i} searches ({sum(nfm_searches.values())} non-matching)\n" \
      f"{len(searches)} unique queries\n\n" \
      f"Most common searches:\n" \
      f"{searches.most_common(5)}\n\n" \
      f"Most common non-full-matching searches:\n" \
      f"{nfm_searches.most_common(5)}\n\n" \
      f"Most common results:\n" \
      f"{results.most_common(5)}\n\n" \
      f"Most vague searches:\n" \
      f"{vague[:5]}\n\n" \
      f"Longest queries:\n" \
      f"{long[:5]}"

print(rpt)

with open(f"stats/out/{outfile}.txt", 'w') as f:
    f.write(rpt)

with open(f"stats/out/results/{outfile}.json", 'w') as f:
    sorted_results = {k: v for k, v in sorted(results.items(), key=lambda p: p[1], reverse=True)}
    json.dump(sorted_results, f, indent=2)

with open(f"stats/out/searches/{outfile}.json", 'w') as f:
    sorted_searches = {k: v for k, v in sorted(searches.items(), key=lambda p: p[1], reverse=True)}
    json.dump(sorted_searches, f, indent=2)

with open(f"stats/out/vagueness/{outfile}.json", 'w') as f:
    sorted_vagueness = {k: list(v) for k, v in sorted(vagueness.items(), key=lambda p: len(p[1]), reverse=True)}
    json.dump(sorted_vagueness, f, indent=2)

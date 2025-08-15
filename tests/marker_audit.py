from reader import read_ptu
from collections import Counter

f = "data\\3mW_100kHzSine.ptu"   
_, _, markers, _ = read_ptu(f)
cnt = Counter()
for mk, _t in markers:
    for b in range(4):
        if mk & (1<<b): cnt[b]+=1
print("per-bit counts:", dict(cnt), " total marker records:", len(markers))

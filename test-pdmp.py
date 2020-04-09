import json
import pdmp
from pathlib import Path


with open('python-objects-osx.json', 'rb') as f:
    objs = json.load(f)

db = pdmp.LibclangDatabase.from_json(objs)

o = Path('a')
p = pdmp.pdmp(o)

p.dump(db, int)

with p.load(db.allocation_report.static_report) as q:
    print(q)

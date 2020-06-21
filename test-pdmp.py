import json
import os
import pdmp
from pathlib import Path


with open('python-objects-osx.json', 'rb') as f:
    objs = json.load(f)

db = pdmp.LibclangDatabase.from_json(objs, Path('libpython3.10.dylib'))

o = Path('a')
p = pdmp.pdmp(o)

# p.dump(db, int)
p.dump(db, 3)

with p.load(db.allocation_report.static_report) as q:
    print(q)
    # print(q)
    # print(q(3))
    # FIXME: we need to use os._exit() instead of sys.exit(), as sys.exit runs some destructors as
    # it unwinds which will reach into memory allocated by the mmap region, but after we close our
    # handle to it!
    os._exit(0)

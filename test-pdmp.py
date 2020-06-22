import json
import logging
import os
import pdmp
from pathlib import Path


logger = logging.getLogger(__name__)


with open('python-objects-osx.json', 'rb') as f:
    objs = json.load(f)

db = pdmp.LibclangDatabase.from_json(objs, Path('libpython3.10.dylib'))

o = Path('a')
p = pdmp.pdmp(o)

class C:
    x = 3

# p.dump(db, C)
p.dump(db, int)
print(f'obj={int}, id={id(int)}')
# p.dump(db, 3)

# TODO:
# 1. make print_python_objects.py into Lib/internal_struct_layout.py (and add a quick test!)! If
#    possible, abstract away the use of libclang specifically, and then put the libclang impl in a
#    script!
# 2. add testing for the expected pdmp file content for 3, int, C, etc (this should fix the
#    segfault!)!

with p.load(db.allocation_report.static_report) as q:
    logger.warning('segfault!')
    print(q(3))
    logger.warning('wow!')
    # print(q)
    # print(q(3))
    # FIXME: we need to use os._exit() instead of sys.exit(), as sys.exit runs some destructors as
    # it unwinds which will reach into memory allocated by the mmap region, but after we close our
    # handle to it!
    os._exit(0)

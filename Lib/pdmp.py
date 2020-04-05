"""
    pdmp
    ~~~~~~~~~

    Methods that allow interacting with a `pdmp` type.
"""

import errno
import fcntl
import gc
import mmap
import os
import pickle
import struct
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from queue import Queue
from typing import Any, Dict, List, Optional, Tuple


_SIZE_BYTES_LONG = 8


def _pack_n_longs(*longs: Tuple[int, ...]) -> bytes:
    return struct.pack('l' * len(longs), *longs)


def _unpack_n_longs(file_handle, n: int) -> Tuple[int, ...]:
    return struct.unpack('l' * n, file_handle.read(_SIZE_BYTES_LONG * n))


@dataclass(frozen=True)
class DumpRelocationEntry:
    source_object: Any
    base_offset: int

    @property
    def volatile_memory_id(self) -> int:
        return id(self.source_object)

    def __post_init__(self) -> None:
        assert self.base_offset >= 0

    @cached_property
    def extent(self) -> int:
        return sys.getsizeof(self.source_object)

    @cached_property
    def referents(self) -> List[Any]:
        return list(gc.get_referents(self.source_object))

    def as_relocation_entry(self) -> bytes:
        return _pack_n_longs(
            self.volatile_memory_id, self.base_offset, self.extent, len(self.referents),
            *[id(obj) for obj in self.referents])

    def as_relocatable_bytes(self) -> bytes:
        return gc.pdmp_write_relocatable_object(self.source_object, self.extent)


@dataclass(frozen=True)
class LoadRelocationEntry:
    original_volatile_memory_id: int
    base_offset: int
    extent: int
    referents: List[int]
    source_object: Any


class Obarray:
    """Named after emacs's `obarray` type, which stores the interned symbol table."""

    def __init__(self) -> None:
        self._obarray: Dict[int, DumpRelocationEntry] = {}

    def put(self, entry: DumpRelocationEntry) -> bool:
        already_exists = entry.volatile_memory_id in self._obarray
        if not already_exists:
            self._obarray[entry.volatile_memory_id] = entry
        return already_exists


@dataclass
class pdmp:
    file_handle: Any
    mmap: Optional[mmap.mmap]

    @property
    def fd(self) -> int:
        return self.file_handle.fileno()

    @property
    def dump_filename(self) -> Path:
        return Path(self.file_handle.name)

    @property
    def mapped_memory_length(self) -> int:
        return os.stat(self.fd).st_size

    class Error(Exception):
        """???"""

    def __init__(self, file_handle) -> None:
        assert file_handle.seekable(), f'{file_handle=} was not seekable!'
        self.file_handle = file_handle
        self.mmap = None

    def is_mapped(self) -> bool:
        return (self.mmap is not None) and (not self.mmap.closed)

    def load(self) -> Any:
        assert self.is_mapped(), f'pdmp object {self=} was not mapped when attempting to load root object!'

        self.mmap.seek(0)
        (num_objects_dumped,) = _unpack_n_longs(self.mmap, 1)

        relocation_info_tuples = []
        for _ in range(num_objects_dumped):
            volatile_memory_id, offset, extent, num_refs = _unpack_n_longs(self.mmap, 4)
            referents = _unpack_n_longs(self.mmap, num_refs)
            relocation_info_tuples.append((
                volatile_memory_id, offset, extent, num_refs, list(referents),
            ))

        end_of_entries_offset = self.mmap.tell()

        relocation_entries: List[LoadRelocationEntry] = []
        for volatile_memory_id, offset, extent, num_refs, referents in relocation_info_tuples:
            stored_object_location = end_of_entries_offset + offset
            ### READ OBJECT FROM RAW MEMORY!!! UNSAFE!!!! ###
            import pdb; pdb.set_trace()
            # FIXME: currently fails with "bus error: python.exe" (probably a segfault)!
            stored_object = self.mmap.read_object_at(stored_object_location)
            entry = LoadRelocationEntry(
                original_volatile_memory_id=volatile_memory_id,
                base_offset=offset,
                extent=extent,
                referents=list(referents),
                source_object=stored_object,
            )
            relocation_entries.append(entry)

        return relocation_entries

    def __enter__(self) -> Any:
        # Get a read lock on the current file.
        fcntl.flock(self.fd, fcntl.LOCK_SH)

        flags = mmap.MAP_SHARED
        prot = mmap.PROT_READ
        self.mmap = mmap.mmap(self.fd, self.mapped_memory_length, flags, prot)
        assert self.is_mapped()
        return self

    def __exit__(self, _exc, _val, _tb) -> None:
        self.mmap.close()
        self.mmap = None
        assert not self.is_mapped()

        # Release the read lock on the current file.
        fcntl.flock(self.fd, fcntl.LOCK_UN)

    @contextmanager
    def _acquire_write_flock_nowait(self):
        """Get a write lock on the current file."""
        try:
            fcntl.flock(self.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as e:
            if e.errno == errno.EWOULDBLOCK:
                raise self.Error(f'the file {self.file_handle=} was locked!!!') from e
            raise

        try:
            yield
        finally:
            fcntl.flock(self.fd, fcntl.LOCK_UN)

    def dump(self, source_object: Any) -> List[DumpRelocationEntry]:
        obarray = Obarray()
        dump_queue = Queue()
        dumped_infos: List[DumpRelocationEntry] = []

        dump_queue.put(source_object)

        # Get a list of all the sizes of each transitively referenced object.
        total_offset: int = 0
        while not dump_queue.empty():
            obj = dump_queue.get()

            # Produce a description of the object in memory.
            entry = DumpRelocationEntry(
                source_object=obj,
                base_offset=total_offset,
            )

            # If the object was already seen, do not attempt to register it again.
            if obarray.put(entry):
                continue

            # Add all objects this one references to the stack to search.
            for ref in entry.referents:
                dump_queue.put(ref)

            # Return the list of relocation entries.
            dumped_infos.append(entry)

            # Bump the offset.
            total_offset += entry.extent

        # Write entries to the file.
        with self._acquire_write_flock_nowait():
            self.file_handle.seek(0)

            num_objects_dumped = len(dumped_infos)
            self.file_handle.write(struct.pack('l', num_objects_dumped))

            for info in dumped_infos:
                self.file_handle.write(info.as_relocation_entry())
            for info in dumped_infos:
                self.file_handle.write(info.as_relocatable_bytes())

        return dumped_infos

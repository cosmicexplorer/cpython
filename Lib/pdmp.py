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
from pathlib import Path
from queue import Queue
from typing import Any, Dict, List, Optional


_SIZE_BYTES_LONG = 8


@dataclass(frozen=True)
class DumpRelocationEntry:
    source_object: Any
    base_offset: int
    referents: List[Any]

    def __post_init__(self) -> None:
        assert self.base_offset >= 0

    @property
    def extent(self) -> int:
        return sys.getsizeof(self.source_object)

    def as_relocation_entry(self) -> bytes:
        return struct.pack('ll', self.base_offset, self.extent)

    def as_relocatable_bytes(self) -> bytes:
        return gc.pdmp_write_relocatable_object(self.source_object, self.extent)


class Obarray:
    """Named after emacs's `obarray` type, which stores the interned symbol table."""

    def __init__(self) -> None:
        self._obarray: Dict[int, Any] = {}

    def put(self, obj: Any) -> bool:
        already_exists = id(obj) in self._obarray
        if not already_exists:
            self._obarray[id(obj)] = obj
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

        num_objects_dumped, = struct.unpack('l', self.mmap.read(_SIZE_BYTES_LONG))

        for _ in range(num_objects_dumped):
            offset, extent = struct.unpack('ll', self.mmap.read(_SIZE_BYTES_LONG * 2))
            # TODO: when reading out each entry, it should also contain locators for the list of
            # entries that *should* be returned by gc.get_referents(obj) for each obj returned by
            # this load()!
            # TODO: We should then overwrite `Py_TYPE(obj)->tp_traverse` on every loaded object so
            # that it specifically traverses over the other located entries within the mmaped
            # region.

        end_of_entries_offset = self.mmap.tell()

    def __enter__(self) -> Any:
        # Get a read lock on the current file.
        fnctl.flock(self.fd, fcntl.LOCK_SH)

        flags = mmap.MAP_FILE | mmap.MAP_SHARED
        prot = mmap.PROT_READ
        access = mmap.ACCESS_READ
        self.mmap = mmap.mmap(self.fd, self.mapped_memory_length, flags, prot, access)
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
            if obarray.put(obj):
                continue

            # Produce a description of the object in memory.
            entry = DumpRelocationEntry(
                source_object=obj,
                base_offset=total_offset,
                referents=gc.get_referents(obj),
            )

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

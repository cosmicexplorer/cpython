"""
    pdmp
    ~~~~~~~~~

    Methods that allow interacting with a `pdmp` type.
"""

import ctypes
import gc
import mmap
import os
import pickle
import struct
import sys
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from functools import cached_property
from pathlib import Path
from queue import Queue
from typing import Any, Dict, List, Optional, Tuple


_SIZE_BYTES_LONG = 8


def _pack_n_longs(*longs: Tuple[int, ...]) -> bytes:
    return struct.pack('l' * len(longs), *longs)


def _unpack_n_longs(file_handle, n: int) -> Tuple[int, ...]:
    return struct.unpack('l' * n, file_handle.read(_SIZE_BYTES_LONG * n))


class ByteConsistencyError(Exception):
    pass


@dataclass(frozen=True)
class RelocationTable:
    mmap: mmap.mmap
    end_of_entries_offset: int

    def _seek_to_entry(self, entry: 'DumpRelocationEntry') -> None:
        self.mmap.seek(self.end_of_entries_offset + entry.base_offset)

    def get_relocated_bytes(self, entry: 'DumpRelocationEntry') -> bytes:
        self._seek_to_entry(entry)
        return self.mmap.read(entry.extent)

    def set_relocated_bytes(self, entry: 'DumpRelocationEntry', new_bytes: bytes) -> None:
        if len(new_bytes) != entry.extent:
            raise ByteConsistencyError(f'new relocated bytes {new_bytes} must be the correct size for entry {entry} (was: {len(new_bytes)})!')
        self._seek_to_entry(entry)
        written = self.mmap.write(new_bytes)
        if written != entry.extent:
            raise ByteConsistencyError(f'wrote fewer bytes ({written}) than expected ({entry.extent}) when updating a pdmp object relocation {entry}!')


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

    def pack_relocation_entry_bytes(self) -> bytes:
        return _pack_n_longs(
            self.volatile_memory_id, self.base_offset, self.extent, len(self.referents),
            *[id(obj) for obj in self.referents])

    def as_volatile_bytes(self) -> bytes:
        return gc.pdmp_write_relocatable_object(self.source_object, self.extent)


@dataclass(frozen=True)
class LoadRelocationEntry:
    original_volatile_memory_id: int
    base_offset: int
    extent: int
    referents: Tuple[int, ...]

    def read_source_object_bytes(self, file_handle) -> bytes:
        return file_handle.read(self.extent)


class Obarray:
    """Named after emacs's `obarray` type, which stores the interned symbol table."""

    def __init__(self) -> None:
        self._obarray: Dict[int, DumpRelocationEntry] = {}

    def put(self, entry: DumpRelocationEntry) -> bool:
        already_exists = entry.volatile_memory_id in self._obarray
        if not already_exists:
            self._obarray[entry.volatile_memory_id] = entry
        return already_exists


@dataclass(frozen=True)
class pdmp:
    file_path: Path

    def _get_mapped_memory_length(self) -> int:
        return os.stat(self.file_path).st_size

    def load(self) -> Any:
        with self._read_handle() as read_mmap:
            read_mmap.seek(0)
            (num_objects_dumped,) = _unpack_n_longs(read_mmap, 1)

            relocation_entries: List[LoadRelocationEntry] = []
            for _ in range(num_objects_dumped):
                volatile_memory_id, offset, extent, num_refs = _unpack_n_longs(read_mmap, 4)
                referents = _unpack_n_longs(read_mmap, num_refs)
                entry = LoadRelocationEntry(
                    original_volatile_memory_id=volatile_memory_id,
                    base_offset=offset,
                    extent=extent,
                    referents=referents,
                )
                relocation_entries.append(entry)

            end_of_entries_offset = read_mmap.tell()

            return relocation_entries

    @contextmanager
    def _read_handle(self):
        file_handle = open(self.file_path, 'rb')

        flags = mmap.MAP_PRIVATE
        prot = mmap.PROT_READ
        length = self._get_mapped_memory_length()
        mapping = mmap.mmap(file_handle.fileno(), length, flags, prot)

        try:
            yield mapping
        finally:
            mapping.close()
            file_handle.close()

    @contextmanager
    def _acquire_write_file_handle(self, *, truncate: bool = True):
        """Get a write lock on the current file."""
        mode = 'wb' if truncate else 'r+b'
        file_handle = open(self.file_path, mode)

        try:
            file_handle.seek(0)
            yield file_handle
        finally:
            file_handle.close()

    @contextmanager
    def _open_write_mapping(self, fd: int):
        flags = mmap.MAP_PRIVATE
        prot = mmap.PROT_WRITE
        length = self._get_mapped_memory_length()
        mapping = mmap.mmap(fd, length, flags, prot)

        try:
            yield mapping
        finally:
            mapping.close()

    def _dump_volatile_memory(
            self,
            file_handle,
            source_object: Any,
    ) -> Tuple[int, bytes, List[DumpRelocationEntry]]:
        obarray = Obarray()
        dump_queue = Queue()
        dump_entries: List[DumpRelocationEntry] = []

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
            dump_entries.append(entry)

            # Bump the offset.
            total_offset += entry.extent

        # Write entries to the pdmp file.
        num_objects_dumped = len(dump_entries)
        file_handle.write(_pack_n_longs(num_objects_dumped))

        for entry in dump_entries:
            file_handle.write(entry.pack_relocation_entry_bytes())

        end_of_entries_offset = file_handle.tell()

        for entry in dump_entries:
            file_handle.write(entry.as_volatile_bytes())

        return (end_of_entries_offset, dump_entries)

    def dump(self, source_object: Any) -> List[DumpRelocationEntry]:
        # Write entries to the file.
        with self._acquire_write_file_handle(truncate=True) as file_handle:
            end_of_entries_offset, dump_entries = self._dump_volatile_memory(file_handle, source_object)

        with self._acquire_write_file_handle(truncate=False) as file_handle,\
             self._open_write_mapping(file_handle.fileno()) as mapping:
                relocation_table = RelocationTable(
                    mmap=mapping,
                    end_of_entries_offset=end_of_entries_offset,
                )
                for entry in dump_entries:
                    relocated_bytes = relocation_table.get_relocated_bytes(entry)
                    import pdb; pdb.set_trace()

        return dump_entries


@dataclass(frozen=True)
class PythonTypeName:
    name: str


@dataclass(frozen=True)
class NativeTypeName:
    name: str


@dataclass(frozen=True)
class FieldName:
    name: str


class ObjectType(Enum):
    plain_old_data = 'plain-old-data'
    pointer = 'pointer'


@dataclass(frozen=True)
class ObjectField:
    object_type: ObjectType
    # the PySetObject's `.smalltable` field allocates some stack space for things, with a
    # statically-known length. The `.table` field is then set to point to `.smalltable` for small
    # sets.
    # However, as in PyBytesObject, `.ob_sval[1]` will always be pointing to something with
    # `.ob_size` elements. So we will want to look that one up in the allocation database we build
    # in `pymalloc_alloc` first, before assuming it's just size 1, for example.
    intrusive_length_hint: Optional[int]


@dataclass(frozen=True)
class LibclangNativeObjectDescriptor:
    native_type_name: NativeTypeName
    fields: OrderedDict  # [FieldName, ObjectField]

    @classmethod
    def from_object(cls, source_object: Any) -> 'LibclangNativeObjectDescriptor':
        raise NotImplementedError


@dataclass(frozen=True)
class LibclangHandle:
    lib: ctypes.CDLL
    cpython_checkout: Path

    def __post_init__(self) -> None:
        assert self.cpython_checkout.is_absolute(), f'path to cpython checkout at {self.cpython_checkout} was not absolute!'

    @cached_property
    def cx_index(self) -> ctypes.c_void_p:
        return self.lib.clang_createIndex(
            ctypes.c_int(0),    # excludeDeclarationsFromPCH
            ctypes.c_int(1),    # displayDiagnostics
        )

    @cached_property
    def _include_dirs(self) -> List[Path]:
        return [
            Path('/usr/local/opt/llvm/include'),
            Path('/usr/include'),
            self.cpython_checkout,
            self.cpython_checkout / 'Include',
            self.cpython_checkout / 'Objects',
            self.cpython_checkout / 'Python',
            self.cpython_checkout / 'Doc' / 'includes',
        ]

    @cached_property
    def _clang_args(self) -> List[str]:
        return [
            '-x', 'c',
            *[f'-I{d}' for d in self._include_dirs]
        ]

    @classmethod
    def _encode_path(cls, path: Path) -> ctypes.c_char_p:
        return ctypes.c_char_p(str(path).encode('ascii'))

    @cached_property
    def _clang_args_ctypes(self):
        ArgsType = ctypes.c_char_p * len(self._clang_args)
        args = ArgsType(*[
            ctypes.c_char_p(arg.encode('ascii'))
            for arg in self._clang_args
        ])
        return (args, len(self._clang_args))

    def tu_from_source_file(self, source_file: Path):
        assert not source_file.is_absolute(), f'C source file path must be relative to cpython checkout at {self.cpython_checkout}! was: {source_file}'
        full_path = self.cpython_checkout / source_file

        clang_args, num_clang_args = self._clang_args_ctypes

        return self.lib.clang_parseTranslationUnit(
            self.cx_index,                      # CIdx
            self._encode_path(full_path),       # source_filename
            clang_args,                         # command_line_args
            ctypes.c_int(num_clang_args),       # num_command_line_args
            None,                               # unsaved_files
            ctypes.c_int(0),                    # num_unsaved_files
            ctypes.c_uint(0),                   # options
        )


@dataclass(frozen=True)
class LibclangDatabase:
    known_objects: OrderedDict  # [PythonTypeName, LibclangNativeObjectDescriptor]

    @classmethod
    def from_file(cls) -> 'LibclangDatabase':
        raise NotImplementedError

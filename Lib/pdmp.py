"""
    pdmp
    ~~~~~~~~~

    Methods that allow interacting with a `pdmp` type.
"""

import gc
import mmap
import os
import re
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
class DumpRelocationEntry:
    source_object: Any

    @property
    def volatile_memory_id(self) -> int:
        return id(self.source_object)

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

    def dump(self, db: 'LibclangDatabase': source_object: Any) -> List[DumpRelocationEntry]:
        with self._acquire_write_file_handle(truncate=True) as file_handle:
            dump_entries = db.dump(relocation_table, source_object)

        return dump_entries


class Spellable(object):

    @property
    def spelling(self) -> str:
        return self._name


@dataclass(frozen=True)
class PythonTypeName(Spellable):
    _name: str


@dataclass(frozen=True)
class NativeTypeName(Spellable):
    _name: str


@dataclass(frozen=True)
class FieldName(Spellable):
    _name: str


class FieldType(Enum):
    plain_old_data = 'plain-old-data'
    pointer = 'pointer'
    array = 'array'
    function_pointer = 'function-pointer'


@dataclass(frozen=True)
class FieldOffset:
    _offset: int

    def as_offset_in_bits(self) -> int:
        return self._offset


@dataclass(frozen=True)
class RecordSize:
    _size: int

    def as_size_in_bits(self) -> int:
        return self._size


@dataclass(frozen=True)
class IntrusiveLength:
    """Length which may be allocate as a fixed-size array in a field of a struct.

    (e.g.) the PySetObject's `.smalltable` field allocates some stack space for things, with a
    statically-known length. The `.table` field is then set to point to `.smalltable` for small
    sets.
    However, in PyBytesObject, `.ob_sval[1]` will always be pointing to something with
    `.ob_size` elements. So we will want to look that one up in the allocation database we build
    in `pymalloc_alloc` first, before assuming it's just size 1, for example.
    """
    length: int


@dataclass(frozen=True)
class ObjectField:
    field_name: FieldName
    object_type: FieldType
    type_name: NativeTypeName
    offset: FieldOffset
    record_size: Optional[RecordSize]
    intrusive_length_hint: Optional[IntrusiveLength]

    @classmethod
    def from_json(cls, input_json: Dict[str, Any]) -> 'ObjectField':
        field_name = FieldName(input_json['field_name'])

        if input_json['is_array']:
            object_type = FieldType.array
        elif input_json['is_function_pointer']:
            object_type = FieldType.function_pointer
        elif input_json['is_pointer']:
            object_type = FieldType.pointer
        else:
            object_type = FieldType.plain_old_data

        intrusive_length_hint = IntrusiveLength(input_json['array_size_hint']) if object_type == FieldType.array else None

        if object_type == FieldType.function_pointer:
            type_name = NativeTypeName(input_json['type_name'])
            record_size = None
        else:
            # FIXME: HACK! this is in case any struct prefixes leak over from
            # print-python-objects.py!!
            type_name = NativeTypeName(re.sub(r'^struct ', '', input_json['scrubbed_type_name']))
            record_size = RecordSize(input_json['non_pointer_record_size'])

        offset = FieldOffset(input_json['offset'])

        return cls(
            field_name=field_name,
            object_type=object_type,
            type_name=type_name,
            offset=offset,
            record_size=record_size,
            intrusive_length_hint=intrusive_length_hint,
        )


@dataclass(frozen=True)
class LibclangNativeObjectDescriptor:
    native_type_name: NativeTypeName
    record_size: RecordSize
    fields: OrderedDict  # [FieldName, ObjectField]

    @classmethod
    def from_json(cls, input_json: Dict[str, Any]) -> 'LibclangNativeObjectDescriptor':
        native_type_name = NativeTypeName(input_json['type_name'])
        record_size = RecordSize(input_json['record_size'])
        fields = OrderedDict([
            (FieldName(entry['field_name']), ObjectField.from_json(entry))
            for entry in input_json['fields']
        ])
        return cls(native_type_name=native_type_name, record_size=record_size, fields=fields)


@dataclass(frozen=True)
class LibclangDatabase:
    ob_type_offset: int
    tp_name_offset: int
    struct_field_mapping: Dict[NativeTypeName, LibclangNativeObjectDescriptor]
    py_object_mapping: Dict[PythonTypeName, NativeTypeName]

    @classmethod
    def from_json(cls, input_json: Dict[str, Any]) -> 'LibclangDatabase':
        ob_type_offset = input_json['ob_type_offset']
        tp_name_offset = input_json['tp_name_offset']

        struct_field_mapping = {
            NativeTypeName(name): LibclangNativeObjectDescriptor.from_json(entry)
            for name, entry in input_json['struct_field_mapping'].items()
        }
        py_object_mapping = {
            PythonTypeName(python_name): NativeTypeName(entry['py_object_struct_name'])
            for python_name, entry in input_json['py_object_mapping'].items()
        }

        return cls(
            ob_type_offset=ob_type_offset,
            tp_name_offset=tp_name_offset,
            struct_field_mapping=struct_field_mapping,
            py_object_mapping=py_object_mapping,
        )

    def dump(self, source_object: Any) -> List[DumpRelocationEntry]:
        pass

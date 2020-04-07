"""
    pdmp
    ~~~~~~~~~

    Methods that allow interacting with a `pdmp` type.
"""

import dataclasses
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
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union


_SIZE_BYTES_LONG = 8


def _pack_n_longs(*longs: Tuple[int, ...]) -> bytes:
    return struct.pack('l' * len(longs), *longs)


def _unpack_n_longs(file_handle, n: int) -> Tuple[int, ...]:
    return struct.unpack('l' * n, file_handle.read(_SIZE_BYTES_LONG * n))


class ByteConsistencyError(Exception):
    pass


@dataclass(frozen=True)
class LoadRelocationEntry:
    original_volatile_memory_id: int
    base_offset: int
    extent: int
    referents: Tuple[int, ...]

    def read_source_object_bytes(self, file_handle) -> bytes:
        return file_handle.read(self.extent)


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

    def dump(self, db: 'LibclangDatabase': source_object: Any) -> List[Any]:
        reachable_objects = db.traverse_reachable_objects(source_object)
        return reachable_objects


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

    @cached_property
    def pointer_depth(self) -> 'PointerDepth':
        return PointerDepth(len(re.findall(r'\*', self.spelling)))

    def dereference_one_pointer_level(self) -> 'NativeTypeName':
        if self.pointer_depth.depth < 1:
            raise ValueError(f'Cannot dereference a non-pointer! Was: {self=}')
        new_name = re.sub(r'\*$', '', self._name)
        assert new_name != self._name, f'Attempting to dereference one pointer level did not change type name: {self=}'
        return dataclasses.replace(self, _name=new_name)


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
class PointerDepth:
    """The number of *s in the type name."""
    depth: int


@dataclass(frozen=True)
class LibclangNativeObjectDescriptor:
    native_type_name: NativeTypeName
    record_size: RecordSize
    pointer_type: PointerType
    fields: OrderedDict  # [FieldName, ObjectField]
    intrusive_length_hint: Optional[IntrusiveLength]

    @classmethod
    def from_json(cls, input_json: Dict[str, Any]) -> 'LibclangNativeObjectDescriptor':
        native_type_name = NativeTypeName(input_json['type_name'])
        record_size = RecordSize(input_json['record_size'])
        fields = OrderedDict([
            (FieldName(entry['field_name']), ObjectField.from_json(entry))
            for entry in input_json['fields']
        ])
        return cls(
            native_type_name=native_type_name,
            record_size=record_size,
            # NB: All struct types that we directly extract from libclang are non-pointer types!
            pointer_type=PointerType.non_pointer,
            fields=fields)


class PythonCLevelType(Enum):
    non_struct = 'non-struct'
    struct = 'struct'


@dataclass(frozen=True)
class PythonCLevelPointerLocation:
    pointer_location: int

    @classmethod
    def from_python_object(cls, source_object: Any) -> 'PythonCLevelPointerLocation':
        return cls(id(source_object))

    @classmethod
    def from_bytes(cls, byte_str: bytes) -> 'PythonCLevelPointerLocation':
        (num_bytes,) = struct.unpack('l', byte_str)
        return cls(num_bytes)

    def __add__(self, other: Any) -> 'PythonCLevelPointerLocation':
        assert isinstance(other, FieldOffset)
        return type(self)(self.pointer_location + other.as_offset_in_bits())


class PointerType(Enum):
    non_pointer = 'non-pointer'
    pointer = 'pointer'
    function_pointer = 'function-pointer'

    @classmethod
    def from_field_type(cls, field_type: FieldType) -> 'PointerType':
        if field_type == FieldType.plain_old_data:
            return cls.non_pointer
        if field_type in [FieldType.pointer, FieldType.array]:
            return cls.pointer
        assert field_type == FieldType.function_pointer
        return cls.function_pointer


@dataclass(frozen=True)
class BasicCLevelTypeDescriptor:
    """E.g. 'int'."""
    native_type_name: NativeTypeName
    record_size: Optional[RecordSize]
    pointer_type: PointerType
    intrusive_length_hint: Optional[IntrusiveLength]


@dataclass
class LivePythonCLevelObject:
    _type_type: PythonCLevelType
    _struct_type_descriptor: Optional[LibclangNativeObjectDescriptor]
    _non_struct_type_descriptor: Optional[BasicCLevelTypeDescriptor]
    _id: PythonCLevelPointerLocation
    _record_size: RecordSize
    _bytes: bytes

    @classmethod
    def from_object(
        cls,
        source_object: Any,
        descriptor: LibclangNativeObjectDescriptor,
    ) -> 'LivePythonCLevelObject':
        return cls.create(
            id=PythonCLevelPointerLocation.from_python_object(source_object),
            descriptor=descriptor,
        )

    @classmethod
    def create(
        cls,
        id: PythonCLevelPointerLocation,
        descriptor: Union[LibclangNativeObjectDescriptor, BasicCLevelTypeDescriptor],
    ) -> 'LivePythonCLevelObject':
        assert isinstance(id, PythonCLevelPointerLocation)
        assert isinstance(record_size, RecordSize)
        object_bytes = gc.pdmp_write_relocatable_object(
            id.pointer_location,
            descriptor.record_size.as_size_in_bits(),
        )

        if isinstance(descriptor, LibclangNativeObjectDescriptor):
            type_type = PythonCLevelType.struct
            struct_type_descriptor = descriptor
            non_struct_type_descriptor = None
        else:
            assert isinstance(descriptor, BasicCLevelTypeDescriptor)
            type_type = PythonCLevelType.non_struct
            struct_type_descriptor = None
            non_struct_type_descriptor = descriptor

        return cls(
            _type_type=type_type,
            _struct_type_descriptor=struct_type_descriptor,
            _non_struct_type_descriptor=non_struct_type_descriptor,
            _id=id,
            _record_size=descriptor.record_size,
            _bytes=object_bytes,
        )

    @property
    def _descriptor(self) -> Union[LibclangNativeObjectDescriptor, BasicCLevelTypeDescriptor]:
        if self._type_type == PythonCLevelType.struct:
            return self._struct_type_descriptor
        return self._non_struct_type_descriptor

    @property
    def pointer_type(self) -> PointerType:
        return self._descriptor.pointer_type

    @property
    def pointer_depth(self) -> PointerDepth:
        return self._descriptor.native_type_name.pointer_depth

    @property
    def volatile_memory_id(self) -> PythonCLevelPointerLocation:
        return self._id

    def get_bytes(self) -> bytes:
        return self._bytes

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, type(self)):
            return False
        return self.volatile_memory_id == other.volatile_memory_id

    def __hash__(self) -> int:
        return hash(self.volatile_memory_id)

    def get_subobjects(self, db: 'LibclangDatabase') -> List['LivePythonCLevelObject']:
        if self.pointer_type == PointerType.function_pointer:
            return []

        if self.pointer_type == PointerType.pointer:
            # TODO: incorporate array size hint / dynamic allocation info from pymalloc_alloc to
            # know how many values to return here!!!
            new_descriptor = dataclasses.replace(
                self._descriptor,
                native_type_name=self._descriptor.native_type_name.dereference_one_pointer_level(),
            )
            return [type(self).create(
                id=PythonCLevelPointerLocation.from_bytes(self.get_bytes()),
                descriptor=new_descriptor,
            )]

        assert self.pointer_type == PointerType.non_pointer
        if self._type_type == PythonCLevelType.non_struct:
            return []
        assert self._type_type == PythonCLevelType.struct

        sub_objects = []
        for field in self._struct_type_descriptor.values():
            descriptor = db.struct_field_mapping.get(field.type_name, None)
            if not descriptor:
                # A struct definition was not found, so this is a "basic" type.
                descriptor = BasicCLevelTypeDescriptor(
                    native_type_name=field.type_name,
                    record_size=field.record_size,
                    pointer_type=PointerType.from_field_type(field.object_type),
                    intrusive_length_hint=field.intrusive_length_hint,
                )
            sub = type(self).create(
                id=(self.volatile_memory_id + field.offset),
                descriptor=descriptor,
            )
            sub_objects.append(sub)
        return sub_objects


class Obarray:
    """Named after emacs's `obarray` type, which stores the interned symbol table."""

    def __init__(self) -> None:
        self._obarray: Set[LivePythonCLevelObject] = set()

    def put(self, entry: LivePythonCLevelObject) -> bool:
        already_exists = entry.volatile_memory_id in self._obarray
        if not already_exists:
            self._obarray.add(entry)
        return already_exists


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

    class InvalidSourceObjectError(Exception):
        pass

    def traverse_reachable_objects(self, source_object: Any) -> List[LivePythonCLevelObject]:
        obarray = Obarray()
        stack = []

        py_type_name = type(source_object).__name__
        # Get the libclang description of the C struct representing the object.
        try:
            py_struct_type = self.py_object_mapping[py_type_name]
        except KeyError as e:
            raise self.InvalidSourceObjectError(
                f'failed to locate libclang entry for {py_type_name}'
            ) from e

        live_entrypoint_object = LivePythonCLevelObject.from_object(source_object, py_struct_type)

        stack.append(live_entrypoint_object)

        reachable_objects: List[LivePythonCLevelObject] = []
        while stack:
            live_object = stack.pop()

            # If the object was already seen, do not attempt to register it again.
            if obarray.put(live_object):
                continue

            reachable_objects.append(live_object)

            stack.extend(live_object.get_subobjects())

        return reachable_objects

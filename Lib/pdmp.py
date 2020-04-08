"""
    pdmp
    ~~~~~~~~~

    Methods that allow interacting with a `pdmp` type.
"""

import dataclasses
import gc
import logging
import mmap
import os
import re
import struct
import subprocess
import sys
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from functools import cached_property
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union


logger = logging.getLogger(__name__)


_SIZE_BYTES_LONG = 8
_BITS_PER_BYTE = 8


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
    def _acquire_write_file_handle(self, *, truncate: bool = False):
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

    def dump(self, db: 'LibclangDatabase', source_object: Any) -> 'ObjectClosure':
        object_closure = db.traverse_reachable_objects(source_object)

        object_closure.fixup()

        with self._acquire_write_file_handle() as file_handle:
            object_closure.dump(file_handle)

        return object_closure


class Spellable(object):

    @property
    def spelling(self) -> str:
        return self._name


@dataclass(frozen=True)
class PythonTypeName(Spellable):
    _name: str

    @classmethod
    def from_python_object(cls, source_object: Any) -> 'PythonTypeName':
        return cls(type(source_object).__name__)


@dataclass(unsafe_hash=True)
class NativeTypeName(Spellable):
    _name: str

    def __init__(self, _name: str) -> None:
        self._name = _name.strip()

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

    def __post_init__(self) -> None:
        assert isinstance(self._offset, int) and (self._offset >= 0), f'bad offset was: {self._offset}'

    def as_offset_in_bits(self) -> int:
        return self._offset

    @classmethod
    def from_record_size(cls, record_size: 'RecordSize') -> 'FieldOffset':
        return cls(record_size.as_size_in_bits())

    @classmethod
    def parse_from_objdump_hex(cls, objdump_hex_offset: str) -> 'FieldOffset':
        return cls(int(objdump_hex_offset, base=16))

    def __mul__(self, n: int) -> 'FieldOffset':
        assert isinstance(n, int) and n >= 0
        return type(self)(self._offset * n)

    def __sub__(self, other: 'FieldOffset') -> 'FieldOffset':
        assert isinstance(other, type(self))
        assert self._offset >= other._offset, f'invalid offset subtraction: {self.offset=}, {other._offset=}'
        return type(self)(self._offset - other._offset)


@dataclass(frozen=True)
class RecordSize:
    _size: int

    @classmethod
    def from_python_object(cls, source_object: Any) -> 'RecordSize':
        return cls(sys.getsizeof(source_object))

    @classmethod
    def from_json(cls, json_int: Optional[int]) -> Optional['RecordSize']:
        if json_int is None:
            return None
        assert isinstance(json_int, int), f'{json_int=} was not int'
        if json_int <= 0:
            return None
        return cls(json_int)

    def __post_init__(self) -> None:
        assert isinstance(self._size, int) and self._size > 0

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
    _length: int

    def get_length(self) -> int:
        return self._length


@dataclass(frozen=True)
class ObjectField:
    field_name: FieldName
    object_type: FieldType
    type_name: NativeTypeName
    non_pointer_type_name: NativeTypeName
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
            record_size = RecordSize.from_json(input_json['non_pointer_record_size'])

        try:
            offset = FieldOffset(input_json['offset'])
        except AssertionError as e:
            raise AssertionError(f'failed parsing {input_json=}: {e}') from e

        return cls(
            field_name=field_name,
            object_type=object_type,
            type_name=type_name,
            non_pointer_type_name=NativeTypeName(input_json['non_pointer_type_name']),
            offset=offset,
            record_size=record_size,
            intrusive_length_hint=intrusive_length_hint,
        )


@dataclass(frozen=True)
class PointerDepth:
    """The number of *s in the type name."""
    depth: int

    def __post_init__(self) -> None:
        assert isinstance(self.depth, int) and (self.depth >= 0)


@dataclass(frozen=True)
class LibclangNativeObjectDescriptor:
    native_type_name: NativeTypeName
    record_size: RecordSize
    pointer_type: 'PointerType'
    fields: OrderedDict  # [FieldName, ObjectField]
    intrusive_length_hint: Optional[IntrusiveLength]

    @classmethod
    def from_json(cls, input_json: Dict[str, Any]) -> 'LibclangNativeObjectDescriptor':
        try:
            native_type_name = NativeTypeName(input_json['type_name'])
            record_size = RecordSize.from_json(input_json['record_size'])
            fields = OrderedDict([
                (FieldName(entry['field_name']), ObjectField.from_json(entry))
                for entry in input_json['fields']
            ])
        except AssertionError as e:
            raise AssertionError(f'failed parsing {input_json=}: {e}') from e

        return cls(
            native_type_name=native_type_name,
            record_size=record_size,
            # NB: All struct types that we directly extract from libclang are non-pointer types!
            pointer_type=PointerType.non_pointer,
            fields=fields,
            intrusive_length_hint=None,
        )


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

    def __add__(self, other: FieldOffset) -> 'PythonCLevelPointerLocation':
        assert isinstance(other, FieldOffset)
        return type(self)(self.pointer_location + other.as_offset_in_bits())

    def __sub__(self, other: 'PythonCLevelPointerLocation') -> FieldOffset:
        assert isinstance(other, type(self))
        assert self.pointer_location >= other.pointer_location, f'invalid subtraction: {self=} - {other=}'
        return FieldOffset(self.pointer_location - other.pointer_location)


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
    def from_python_object(
        cls,
        source_object: Any,
        descriptor: Union[LibclangNativeObjectDescriptor, BasicCLevelTypeDescriptor],
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

        record_size = descriptor.record_size
        if record_size is None:
            record_size = RecordSize(_SIZE_BYTES_LONG * _BITS_PER_BYTE)

        object_bytes = gc.pdmp_write_relocatable_object(
            id.pointer_location,
            record_size.as_size_in_bits(),
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
            _record_size=record_size,
            _bytes=object_bytes,
        )

    @property
    def _descriptor(self) -> Union[LibclangNativeObjectDescriptor, BasicCLevelTypeDescriptor]:
        if self._type_type == PythonCLevelType.struct:
            return self._struct_type_descriptor
        return self._non_struct_type_descriptor

    @property
    def record_size(self) -> Optional[RecordSize]:
        return self._descriptor.record_size

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

    @property
    def intrusive_length_hint(self) -> Optional[IntrusiveLength]:
        return self._descriptor.intrusive_length_hint

    def get_subobjects(self, db: 'LibclangDatabase') -> List['LivePythonCLevelObject']:
        if self.pointer_type == PointerType.function_pointer:
            return []

        if self.pointer_type == PointerType.pointer:
            if self.pointer_depth.depth == 1:
                record_size = self.record_size
            else:
                assert self.pointer_depth.depth > 0
            if self.record_size is None:
                record_size = RecordSize(_SIZE_BYTES_LONG * _BITS_PER_BYTE)

            allocation_size = db.allocation_report.get(self.volatile_memory_id)
            if allocation_size is None:
                # If we can't find the allocation, we assume it's intrusive. We require that the
                # intrusive size hint was provided.
                num_allocations = self.intrusive_length_hint
                if num_allocations is None:
                    # import pdb; pdb.set_trace()
                    logger.warn(f'Could not find an intrusive length hint, and allocation was not traced. Assuming 1...')
                    num_allocations = IntrusiveLength(1)
                    return []
            else:
                assert (allocation_size == 1) or (allocation_size % record_size.as_size_in_bits() == 0)
                num_allocations = IntrusiveLength(allocation_size // record_size.as_size_in_bits())

            assert num_allocations.get_length() > 0

            reduced_type_name = self._descriptor.native_type_name.dereference_one_pointer_level()
            new_descriptor = dataclasses.replace(
                self._descriptor,
                native_type_name=reduced_type_name,
                pointer_type=(
                    PointerType.non_pointer if reduced_type_name.pointer_depth.depth == 0
                    else PointerType.pointer
                ),
                intrusive_length_hint=None,
            )
            base_pointer = self.volatile_memory_id

            sub_pointers = [
                type(self).create(
                    id=(base_pointer + (FieldOffset.from_record_size(record_size) * cur_element_index)),
                    descriptor=new_descriptor,
                )
                for cur_element_index in range(num_allocations.get_length())
            ]

            return sub_pointers

        assert self.pointer_type == PointerType.non_pointer
        if self._type_type == PythonCLevelType.non_struct:
            return []
        assert self._type_type == PythonCLevelType.struct

        sub_objects = []
        for field in self._struct_type_descriptor.fields.values():
            descriptor = db.struct_field_mapping.get(field.non_pointer_type_name, None)
            if descriptor:
                descriptor = dataclasses.replace(
                    descriptor,
                    # record_size=field.record_size,
                    native_type_name=field.type_name,
                    pointer_type=PointerType.from_field_type(field.object_type),
                )
            else:
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
class AllocationSize:
    _size: int

    def __post_init__(self) -> None:
        assert self._size >= 0

    def as_size_in_bits(self) -> int:
        return self._size


class StaticSymbolType(Enum):
    text = 'text'
    data = 'data'

    @classmethod
    def parse_from_segment_descriptor(cls, segment_descriptor: str) -> 'StaticSymbolType':
        if segment_descriptor == '__TEXT,__text':
            return cls.text
        if segment_descriptor == '__DATA,__data':
            return cls.data
        raise TypeError(f'{segment_descriptor=} was not a valid static symbol type!')


@dataclass(frozen=True)
class StaticSymbolName:
    name: str


@dataclass(frozen=True)
class LiveStaticSymbol:
    symbol_type: StaticSymbolType
    location: PythonCLevelPointerLocation
    size: Optional[AllocationSize]
    name: StaticSymbolName


@dataclass(frozen=True)
class StaticAllocationReport:
    text_allocation_report: Dict[PythonCLevelPointerLocation, LiveStaticSymbol]
    data_allocation_report: Dict[PythonCLevelPointerLocation, LiveStaticSymbol]

    def get(self, id: PythonCLevelPointerLocation) -> Optional[AllocationSize]:
        if text_entry := self.text_allocation_report.get(id, None):
            return text_entry.size
        if data_entry := self.data_allocation_report.get(id, None):
            return data_entry.size
        return None

    # 0000000100002500 l     F __TEXT,__text  _freechildren
    # 000000010027aa68 l     O __DATA,__data  _empty_keys_struct
    _static_record_pattern = re.compile(r'^([0-9]{16}) l     [FO] (__TEXT__,__text|__DATA,__data)\t(.*)$')

    @classmethod
    def extract_from_executable(cls) -> 'StaticAllocationReport':
        # Get the start of the text and data sections.
        etext, edata = gc.pdmp_get_data_and_text_segment_starts()
        text_segment_start = PythonCLevelPointerLocation(etext)
        data_segment_start = PythonCLevelPointerLocation(edata)

        # Now, extract the records of allocations from the python executable.
        dumped_segments = list(
            subprocess.run(['objdump', '-macho', '-t', '-all-headers', sys.executable],
                           capture_output=True,
                           check=True)
            .stdout
            .decode()
            .splitlines())

        assert dumped_segments[0] == f'{sys.executable}:'
        assert dumped_segments[1] == 'Sections:'
        assert dumped_segments[2].startswith('Idx')

        #   0 __text        001d44ae 0000000100001680 TEXT
        dumped_text_segment = re.match(r'  0 __text        ([0-9a-f]{8}) ([0-9a-f]{16}) TEXT',
                                       dumped_segments[3])
        assert dumped_text_segment is not None
        _dumped_text_size, dumped_text_start = dumped_text_segment.groups()
        dumped_text_start = PythonCLevelPointerLocation(int(dumped_text_start, base=16))
        live_text_segment_offset = text_segment_start - dumped_text_start

        #   9 __data        0003362d 0000000100269980 DATA
        dumped_data_segment = re.match(r'  9 __data        ([0-9a-f]{8}) ([0-9a-f]{16}) DATA',
                                       dumped_segments[12])
        assert dumped_data_segment is not None
        _dumped_data_size, dumped_data_start = dumped_data_segment.groups()
        dumped_data_start = PythonCLevelPointerLocation(int(dumped_data_start, base=16))
        live_data_segment_offset = data_segment_start - dumped_data_start

        assert dumped_segments[16] == 'SYMBOL TABLE:'

        possibly_parseable_static_symbols = dumped_segments[17:]

        text_allocation_report = {}
        data_allocation_report = {}
        for index, line in enumerate(possibly_parseable_static_symbols):
            if static_segment_record := cls._static_record_pattern.match(line):
                hex_offset, symbol_type_str, symbol_name = tuple(static_segment_record.groups())
                symbol_type = StaticSymbolType.parse_from_segment_descriptor(symbol_type_str)
                offset = FieldOffset.parse_from_objdump_hex(hex_offset)

                if symbol_type == StaticSymbolType.text:
                    location = text_segment_start + live_text_segment_offset + offset
                else:
                    assert symbol_type == StaticSymbolType.data
                    location = data_segment_start + live_data_segment_offset + offset

                allocation_size = None
                if index < len(possibly_parseable_static_symbols) - 1:
                    next_line = possibly_parseable_static_symbols[index + 1]
                    if next_record := cls._static_record_pattern.match(next_line):
                        next_offset = FieldOffset.parse_from_objdump_hex(next_record.groups()[0])
                        if next_offset.as_offset_in_bits() != 0:
                            guessed_size = (next_offset - offset).as_offset_in_bits()
                            if guessed_size >= 0:
                                allocation_size = AllocationSize(guessed_size)

                symbol = LiveStaticSymbol(
                    symbol_type=symbol_type,
                    location=location,
                    size=allocation_size,
                    name=StaticSymbolName(symbol_name),
                )

                if symbol_type == StaticSymbolType.text:
                    text_allocation_report[location] = symbol
                else:
                    data_allocation_report[location] = symbol

        return cls(text_allocation_report=text_allocation_report,
                   data_allocation_report=data_allocation_report)


@dataclass(frozen=True)
class RawAllocationReport:
    records: Dict[PythonCLevelPointerLocation, AllocationSize]

    @cached_property
    def static_report(self) -> StaticAllocationReport:
        return StaticAllocationReport.extract_from_executable()

    def get(self, id: PythonCLevelPointerLocation) -> Optional[AllocationSize]:
        if live_record_size := self.records.get(id, None):
            return live_record_size
        if static_record_size := self.static_report.get(id):
            return static_record_size
        return None

    @classmethod
    def get_current_report(cls, db: 'LibclangDatabase') -> 'RawAllocationReport':
        single_record_size = db.allocation_record_struct.record_size.as_size_in_bits()
        all_allocations_all_bytes = gc.pdmp_write_allocation_report()
        assert len(all_allocations_all_bytes) % single_record_size == 0
        num_allocations = len(all_allocations_all_bytes) // single_record_size

        records: Dict[PythonCLevelPointerLocation, AllocationSize] = {}
        for cur_record_index in range(0, num_allocations):
            cur_offset = cur_record_index * single_record_size
            cur_record_bytes = all_allocations_all_bytes[cur_offset:(cur_offset + single_record_size)]
            cur_pointer, cur_nbytes = struct.unpack(
                'PN',
                cur_record_bytes)
            records[PythonCLevelPointerLocation(cur_pointer)] = AllocationSize(cur_nbytes)

        return cls(records)



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

        struct_field_mapping = {}
        for name, entry in input_json['struct_field_mapping'].items():
            try:
                struct_field_mapping[NativeTypeName(name)] = LibclangNativeObjectDescriptor.from_json(entry)
            except AssertionError as e:
                # logger.exception(e)
                continue

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

    @cached_property
    def allocation_record_struct(self) -> LibclangNativeObjectDescriptor:
        return self.struct_field_mapping[NativeTypeName('allocation_record')]

    @cached_property
    def allocation_report(self) -> RawAllocationReport:
        return RawAllocationReport.get_current_report(self)

    class InvalidSourceObjectError(Exception):
        pass

    def traverse_reachable_objects(self, source_object: Any) -> 'ObjectClosure':
        obarray = Obarray()
        stack = []

        py_type_name = PythonTypeName.from_python_object(source_object)
        # Get the libclang description of the C struct representing the object.
        try:
            py_struct_name = self.py_object_mapping[py_type_name]
        except KeyError as e:
            raise self.InvalidSourceObjectError(
                f'failed to locate libclang entry for {py_type_name}'
            ) from e

        try:
            descriptor = self.struct_field_mapping[py_struct_name]
        except KeyError as e:
            # A struct definition was not found, so this is a "basic" type.
            descriptor = BasicCLevelTypeDescriptor(
                native_type_name=py_struct_name,
                record_size=RecordSize.from_python_object(source_object),
                pointer_type=PointerType.non_pointer,
                intrusive_length_hint=None,
            )

        live_entrypoint_object = LivePythonCLevelObject.from_python_object(source_object,
                                                                           descriptor)

        stack.append(live_entrypoint_object)

        reachable_objects: List[LivePythonCLevelObject] = []
        while stack:
            live_object = stack.pop()

            # If the object was already seen, do not attempt to register it again.
            if obarray.put(live_object):
                continue

            reachable_objects.append(live_object)

            stack.extend(live_object.get_subobjects(self))

        return ObjectClosure(reachable_objects)


@dataclass
class ObjectClosure:
    objects: Dict[PythonCLevelPointerLocation, LivePythonCLevelObject]

    def __init__(self, objects: List[LivePythonCLevelObject]) -> None:
        self.objects = {obj.volatile_memory_id: obj for obj in objects}

    def fixup(self) -> None:
        """Within the set of reachable objects, """
        ...

    def dump(self, file_handle) -> int:
        ...

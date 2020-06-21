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
_SIZE_BYTES_POINTER = 8
_BITS_PER_BYTE = 8


def _pack_longs(*longs: Tuple[int, ...]) -> bytes:
    return struct.pack('l' * len(longs), *longs)


def _unpack_n_longs(file_handle, n: int) -> Tuple[int, ...]:
    return struct.unpack('l' * n, file_handle.read(_SIZE_BYTES_LONG * n))


_is_osx = os.uname().sysname == 'Darwin'


class ByteConsistencyError(Exception):
    pass


@dataclass(frozen=True)
class pdmp:
    file_path: Path

    def _get_mapped_memory_length(self) -> int:
        return os.stat(self.file_path).st_size

    @contextmanager
    def load(self, static_report: 'StaticAllocationReport'):
        with self._acquire_write_file_handle(truncate=False) as file_handle:
            ### LET'S READ SOME BYTES!!!
            # (1-2) Read the size of the first object, the etext pointer, and the edata pointer.
            (mmap_start_address, etext_value, edata_value) = struct.unpack(
                'PPP',
                file_handle.read(_SIZE_BYTES_POINTER * 3))

            initial_address = PythonCLevelPointerLocation(mmap_start_address)
            hardcoded_start_address = PythonCLevelPointerLocation(ObjectClosure.MMAP_START_ADDRESS().as_offset_in_bits())
            assert initial_address == hardcoded_start_address, f'{initial_address=} from {self=} did not match hardcoded address {hardcoded_start_address=}'

            etext = PythonCLevelPointerLocation(etext_value)
            assert etext == static_report.text_segment_start, f'etext value from pdump differed: {etext=} vs {static_report.text_segment_start=}'
            edata = PythonCLevelPointerLocation(edata_value)
            assert edata == static_report.data_segment_start, f'edata value from pdump differed: {edata=} vs {static_report.data_segment_start=}'

            with self._open_private_write_mapping(file_handle.fileno(),
                                                  initial_address=initial_address) as write_mapping:
                reloaded_object = write_mapping.read_object_at(file_handle.tell())
                # import pdb; pdb.set_trace()
                yield reloaded_object

    @contextmanager
    def _acquire_write_file_handle(self, *, truncate: bool):
        mode = 'wb' if truncate else 'r+b'
        file_handle = open(self.file_path, mode)

        try:
            yield file_handle
        finally:
            file_handle.close()

    @contextmanager
    def _open_private_write_mapping(self, fd: int, initial_address: 'PythonCLevelPointerLocation'):
        flags = mmap.MAP_PRIVATE
        prot = mmap.PROT_READ | mmap.PROT_WRITE
        length = self._get_mapped_memory_length()
        mapping = mmap.mmap(fileno=fd,
                            length=length,
                            flags=flags,
                            prot=prot,
                            initial_address=initial_address.pointer_location)

        try:
            yield mapping
        finally:
            mapping.close()

    def dump(self, db: 'LibclangDatabase', source_object: Any) -> 'ObjectClosure':
        object_closure = db.traverse_reachable_objects(source_object)

        file_contents = object_closure.dumps(db.allocation_report)
        with self._acquire_write_file_handle(truncate=True) as file_handle:
            file_handle.write(file_contents)

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
        assert isinstance(self._offset, int) and (self._offset >= 0), f'bad offset: {self=}'

    def as_offset_in_bits(self) -> int:
        return self._offset

    @classmethod
    def from_record_size(cls, record_size: 'RecordSize') -> 'FieldOffset':
        return cls(record_size.as_size_in_bits())

    @classmethod
    def parse_from_objdump_hex(cls, objdump_hex_offset: str) -> 'FieldOffset':
        return cls(int(objdump_hex_offset, base=16))

    def __add__(self, arg) -> 'FieldOffset':
        if isinstance(arg, bytes):
            return type(self)(self._offset + len(arg))
        else:
            assert isinstance(self, type(self))
            return type(self)(self._offset + arg._offset)

    def __mul__(self, n: int) -> 'FieldOffset':
        assert isinstance(n, int) and n >= 0
        return type(self)(self._offset * n)

    def __sub__(self, other: 'FieldOffset') -> 'FieldOffset':
        assert isinstance(other, type(self))
        assert self._offset >= other._offset, f'invalid offset subtraction: {self._offset=}, {other._offset=}'
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
    """Length which may be allocated as a fixed-size array in a field of a struct.

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

    def __post_init__(self) -> None:
        assert isinstance(self.pointer_location, int)

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
        allocation_report: 'RawAllocationReport',
    ) -> 'LivePythonCLevelObject':
        return cls.create(
            id=PythonCLevelPointerLocation.from_python_object(source_object),
            descriptor=descriptor,
            allocation_report=allocation_report,
        )

    @classmethod
    def create(
        cls,
        id: PythonCLevelPointerLocation,
        descriptor: Union[LibclangNativeObjectDescriptor, BasicCLevelTypeDescriptor],
        allocation_report: 'RawAllocationReport',
    ) -> 'LivePythonCLevelObject':
        assert isinstance(id, PythonCLevelPointerLocation)

        record_size = descriptor.record_size
        if record_size is None:
            record_size = RecordSize(_SIZE_BYTES_LONG * _BITS_PER_BYTE)

        logger.info(f'attempted read at {id=} of size {record_size=}')
        # import pdb; pdb.set_trace()
        allocation_report.validate_attempted_process_memory_read(
            start=id.pointer_location,
            length=record_size.as_size_in_bits(),
        )

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

    def is_pointer(self) -> bool:
        return self.pointer_depth.depth > 0

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
                record_size = RecordSize(_SIZE_BYTES_LONG * _BITS_PER_BYTE)
            if record_size is None:
                record_size = RecordSize(_SIZE_BYTES_LONG * _BITS_PER_BYTE)

            allocation_record = db.allocation_report.get(self.volatile_memory_id)
            allocation_size = (allocation_record and allocation_record.size.as_size_in_bits())
            if allocation_size is None:
                # If we can't find the allocation, we assume it's intrusive. We require that the
                # intrusive size hint was provided.
                num_allocations = self.intrusive_length_hint
                if num_allocations is None:
                    # num_allocations = IntrusiveLength(1)
                    # num_allocations = IntrusiveLength(len(self.get_bytes()) // _SIZE_BYTES_POINTER)
                    logger.warning(f'could not detect allocation at {self=}: assuming size 0')
                    return []
            else:
                assert (allocation_size == 1) or (allocation_size % record_size.as_size_in_bits() == 0)
                num_allocations = IntrusiveLength(allocation_size // record_size.as_size_in_bits())

            if num_allocations.get_length() < 0:
                # FIXME: hack!!! why is this happening??? we get a length of -1 sometimes??
                num_allocations = IntrusiveLength(num_allocations.get_length() * -1)
            elif num_allocations.get_length() == 0:
                return []

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

            sub_pointers = []
            for cur_element_index in range(num_allocations.get_length()):
                try:
                    subobject = type(self).create(
                        id=(base_pointer + (FieldOffset.from_record_size(record_size) * cur_element_index)),
                        descriptor=new_descriptor,
                        allocation_report=db.allocation_report,
                    )
                    sub_pointers.append(subobject)
                except InvalidAttemptedProcessMemoryRead:
                    continue

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
                    record_size=field.record_size,
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
                allocation_report=db.allocation_report,
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


class AllocationType(Enum):
    static = 'static'
    heap = 'heap'


class StaticSymbolType(Enum):
    text = '.text'
    data = '.data'


@dataclass(frozen=True)
class StaticSymbolName:
    name: str


@dataclass(frozen=True)
class StaticSecondarySymbolType:
    """TODO: what does this part mean??"""
    symbol_type_name: str


@dataclass(frozen=True)
class LiveStaticSymbol:
    symbol_type: StaticSymbolType
    location: PythonCLevelPointerLocation
    size: Optional[AllocationSize]
    secondary: StaticSecondarySymbolType
    name: StaticSymbolName


@dataclass(frozen=True)
class StaticSectionsHeader:
    dumped_text_start: FieldOffset
    dumped_data_start: FieldOffset
    possibly_parseable_static_symbols: List[str]


@dataclass(frozen=True)
class StaticAllocationReport:
    text_segment_start: PythonCLevelPointerLocation
    data_segment_start: PythonCLevelPointerLocation
    text_allocation_report: Dict[PythonCLevelPointerLocation, LiveStaticSymbol]
    data_allocation_report: Dict[PythonCLevelPointerLocation, LiveStaticSymbol]

    @cached_property
    def all_conceivable_allocations(self) -> Dict[PythonCLevelPointerLocation, AllocationSize]:
        return {
            loc: sym.size
            for loc, sym in [
                    *self.text_allocation_report.items(),
                    *self.data_allocation_report.items(),
            ]
            if sym.size is not None
        }

    def get(self, id: PythonCLevelPointerLocation) -> Optional['TrackedAllocation']:
        if text_entry := self.text_allocation_report.get(id, None):
            return TrackedAllocation(
                allocation_type=AllocationType.static,
                size=text_entry.size)
        if data_entry := self.data_allocation_report.get(id, None):
            return TrackedAllocation(
                allocation_type=AllocationType.static,
                size=data_entry.size)
        return None

    @classmethod
    def _parse_static_sections(
            cls,
            path: Path,
            dumped_segments: List[str],
    ) -> StaticSectionsHeader:
        # 0 .text         001ee11e  00000000000007f0  00000000000007f0  000007f0  2**4
        text_segment_pattern = re.compile(r'0 \.text[ \t]+([0-9a-f]+)[ \t]+([0-9a-f]+)')
        # 9 .data         000343bd  00000000002829a0  00000000002829a0  002829a0  2**4
        data_segment_pattern = re.compile(r'9 \.data[ \t]+([0-9a-f]+)[ \t]+([0-9a-f]+)')
        # import pdb; pdb.set_trace()
        for index, line in enumerate(dumped_segments):
            if dumped_text_segment := text_segment_pattern.search(line):
                (_text_size, dumped_text_start,) = dumped_text_segment.groups()
                dumped_text_start = PythonCLevelPointerLocation(int(dumped_text_start, base=16))
            if dumped_data_segment := data_segment_pattern.search(line):
                (_data_size, dumped_data_start,) = dumped_data_segment.groups()
                dumped_data_start = PythonCLevelPointerLocation(int(dumped_data_start, base=16))
            if line == 'SYMBOL TABLE:':
                possibly_parseable_static_symbols = dumped_segments[(index + 1):]
                break

        return StaticSectionsHeader(
            dumped_text_start=dumped_text_start,
            dumped_data_start=dumped_data_start,
            possibly_parseable_static_symbols=possibly_parseable_static_symbols,
        )

    @classmethod
    def _parse_static_segment_record(
        cls,
        line: str,
        text_segment_start: FieldOffset,
        data_segment_start: FieldOffset,
        live_text_segment_offset: FieldOffset,
        live_data_segment_offset: FieldOffset,
        next_line: Optional[str],
    ) -> Optional[LiveStaticSymbol]:
        # 0000000000001ee0 g       1e SECT   01 0000 [.text] __PyPegen_lookahead_with_name
        _static_record_pattern = re.compile(r'^([0-9a-f]{16})[ \t]+.*\[(\.text|\.data)\][ \t]+([a-zA-Z_]+)$')
        if static_segment_record := _static_record_pattern.match(line):
            hex_offset, symbol_type_str, symbol_name = tuple(static_segment_record.groups())
            secondary_name = None

            symbol_type = StaticSymbolType(symbol_type_str)

            offset = FieldOffset.parse_from_objdump_hex(hex_offset)

            if symbol_type == StaticSymbolType.text:
                location = text_segment_start + live_text_segment_offset + offset
            else:
                assert symbol_type == StaticSymbolType.data
            location = data_segment_start + live_data_segment_offset + offset

            allocation_size = None
            if next_line and (next_record := _static_record_pattern.match(next_line)):
                next_offset = FieldOffset.parse_from_objdump_hex(next_record.groups()[0])
                if next_offset.as_offset_in_bits() != 0:
                    try:
                        guessed_size = (next_offset - offset).as_offset_in_bits()
                    except AssertionError:
                        guessed_size = 0
                    if guessed_size >= 0:
                        allocation_size = AllocationSize(guessed_size)
        else:
            return None

        symbol = LiveStaticSymbol(
            symbol_type=symbol_type,
            location=location,
            size=allocation_size,
            secondary=(secondary_name and StaticSecondarySymbolType(secondary_name)),
            name=StaticSymbolName(symbol_name),
        )
        return symbol

    @classmethod
    def extract_from_executable(cls, path: Path) -> 'StaticAllocationReport':
        # Get the start of the text and data sections.
        etext, edata = gc.pdmp_get_data_and_text_segment_starts()
        text_segment_start = PythonCLevelPointerLocation(etext)
        data_segment_start = PythonCLevelPointerLocation(edata)

        # Now, extract the records of allocations from the python executable.

        platform_specific_objdump_arguments = (
            ['--all-headers']
        )

        dumped_segments = list(
            subprocess.run(['objdump', '-t',
                            *platform_specific_objdump_arguments,
                            str(path)],
                           capture_output=True,
                           check=True)
            .stdout
            .decode()
            .splitlines())

        static_sections_header = cls._parse_static_sections(path, dumped_segments)
        live_text_segment_offset = text_segment_start - static_sections_header.dumped_text_start
        live_data_segment_offset = data_segment_start - static_sections_header.dumped_data_start
        possibly_parseable_static_symbols = static_sections_header.possibly_parseable_static_symbols

        text_allocation_report = {}
        data_allocation_report = {}
        for index, line in enumerate(possibly_parseable_static_symbols):
            next_line = None
            if index < len(possibly_parseable_static_symbols) - 1:
                next_line = possibly_parseable_static_symbols[index + 1]
            if symbol := cls._parse_static_segment_record(
                line=line,
                text_segment_start=text_segment_start,
                data_segment_start=data_segment_start,
                live_text_segment_offset=live_text_segment_offset,
                live_data_segment_offset=live_data_segment_offset,
                next_line=next_line,
            ):
                if symbol.symbol_type == StaticSymbolType.text:
                    text_allocation_report[symbol.location] = symbol
                else:
                    data_allocation_report[symbol.location] = symbol

        return cls(
            text_segment_start=text_segment_start,
            data_segment_start=data_segment_start,
            text_allocation_report=text_allocation_report,
            data_allocation_report=data_allocation_report)


@dataclass(frozen=True)
class TrackedAllocation:
    allocation_type: AllocationType
    size: Optional[AllocationSize]


class InvalidAttemptedProcessMemoryRead(Exception):
    """???"""


@dataclass(frozen=True)
class RawAllocationReport:
    records: Dict[PythonCLevelPointerLocation, AllocationSize]
    static_report: StaticAllocationReport

    @cached_property
    def all_conceivable_allocations(self) -> Dict[PythonCLevelPointerLocation, AllocationSize]:
        return {
            **self.static_report.all_conceivable_allocations,
            **self.records,
        }

    @cached_property
    def _all_allocation_starts(self) -> List[int]:
        return list(sorted(
            loc.pointer_location for loc in
            self.all_conceivable_allocations.keys()
        ))

    def validate_attempted_process_memory_read(self, start: int, length: int) -> None:
        assert len(self._all_allocation_starts) > 0
        assert start > 0
        assert length > 0

        if start < self._all_allocation_starts[0]:
            raise InvalidAttemptedProcessMemoryRead(f'{start=} is less than minimum allocation start {self._all_allocation_starts[0]=}')

        greatest_allocation_start = self._all_allocation_starts[0]
        for maybe_start in self._all_allocation_starts:
            if maybe_start > start:
                break
            greatest_allocation_start = maybe_start

        loc = PythonCLevelPointerLocation(greatest_allocation_start)
        allocation_size = self.all_conceivable_allocations[loc]

        allocation_end = loc.pointer_location + allocation_size.as_size_in_bits()
        desired_end = start + length

        # import pdb; pdb.set_trace()
        if desired_end > allocation_end:
            raise InvalidAttemptedProcessMemoryRead(f'{length=} is greater than the known allocated region of size {allocation_size.as_size_in_bits()=}')

    def get(self, id: PythonCLevelPointerLocation) -> Optional[TrackedAllocation]:
        if live_record_size := self.records.get(id, None):
            return TrackedAllocation(
                allocation_type=AllocationType.heap,
                size=live_record_size,
            )
        if static_record := self.static_report.get(id):
            return static_record
        return None

    @classmethod
    def get_current_report(cls, db: 'LibclangDatabase', executable: Path) -> 'RawAllocationReport':
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
            # FIXME: why this particular scaling factor???
            records[PythonCLevelPointerLocation(cur_pointer)] = AllocationSize(cur_nbytes * _BITS_PER_BYTE * 4)

        static_report = StaticAllocationReport.extract_from_executable(executable)

        return cls(records=records, static_report=static_report)



@dataclass(frozen=True)
class LibclangDatabase:
    ob_type_offset: int
    tp_name_offset: int
    struct_field_mapping: Dict[NativeTypeName, LibclangNativeObjectDescriptor]
    py_object_mapping: Dict[PythonTypeName, NativeTypeName]
    allocation_report: RawAllocationReport

    @classmethod
    def from_json(cls, input_json: Dict[str, Any], executable: Path) -> 'LibclangDatabase':
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

        # FIXME: remove this gross hack to get the static allocation report!!
        ret = cls(
            ob_type_offset=ob_type_offset,
            tp_name_offset=tp_name_offset,
            struct_field_mapping=struct_field_mapping,
            py_object_mapping=py_object_mapping,
            allocation_report=None,
        )
        allocation_report = RawAllocationReport.get_current_report(ret, executable)

        return cls(
            ob_type_offset=ob_type_offset,
            tp_name_offset=tp_name_offset,
            struct_field_mapping=struct_field_mapping,
            py_object_mapping=py_object_mapping,
            allocation_report=allocation_report,
        )

    @cached_property
    def allocation_record_struct(self) -> LibclangNativeObjectDescriptor:
        return self.struct_field_mapping[NativeTypeName('allocation_record')]

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

        # import pdb; pdb.set_trace()
        live_entrypoint_object = LivePythonCLevelObject.from_python_object(
            source_object,
            descriptor,
            allocation_report=self.allocation_report)

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


@dataclass(frozen=True)
class RelocatedObject:
    original: LivePythonCLevelObject
    new_offset: FieldOffset


@dataclass
class ObjectClosure:
    _live_object_list: List[LivePythonCLevelObject]
    _objects: Dict[PythonCLevelPointerLocation, LivePythonCLevelObject]

    # _MMAP_ARBITRARY_STARTING_ADDRESS = FieldOffset(2 ** 62)
    _MMAP_ARBITRARY_STARTING_ADDRESS = FieldOffset(2 ** 31)

    @classmethod
    def MMAP_START_ADDRESS(cls) -> FieldOffset:
        return cls._MMAP_ARBITRARY_STARTING_ADDRESS

    def __init__(self, objects: List[LivePythonCLevelObject]) -> None:
        self._live_object_list = objects
        self._objects = {obj.volatile_memory_id: obj for obj in objects}

    def dumps(self, allocation_report: RawAllocationReport) -> bytes:
        """Write the bytes of every non-static object out to file."""

        all_bytes = b''

        # Line up all the heap-allocated python objects consecutively (without writing anything to
        # the file just yet).
        current_offset = FieldOffset(0)
        offsets = OrderedDict() # [PythonCLevelPointerLocation, FieldOffset]

        for obj in self._live_object_list:
            allocation_record = allocation_report.get(obj.volatile_memory_id)
            if allocation_record and (allocation_record.allocation_type == AllocationType.static):
                continue

            # If we can't find the object's allocation, we pretend it was heap-allocated, and we
            # track it. Elsewhere, we assume that pointers we can't locate are sized 1 (as in,
            # allocated as an array of size one of the struct they represent).
            offsets[obj.volatile_memory_id] = current_offset
            current_offset += obj.get_bytes()

        ### LET'S WRITE SOME BYTES!!!
        # (1) Write out the mmap offset that the file should be loaded at!
        all_bytes += struct.pack('P', self.MMAP_START_ADDRESS().as_offset_in_bits())

        # (2) Write out the start of the text and data segments. We will validate that these are the
        # same when we load the pdmp file.
        text_start = allocation_report.static_report.text_segment_start
        all_bytes += struct.pack('P', text_start.pointer_location)
        data_start = allocation_report.static_report.data_segment_start
        all_bytes += struct.pack('P', data_start.pointer_location)

        # (3) Relocate the objects!!
        relocations = OrderedDict() # [PythonCLevelPointerLocation, RelocatedObject]
        object_initial_offset = FieldOffset(len(all_bytes))
        for original_id, basic_offset in offsets.items():
            new_offset = (object_initial_offset + basic_offset)
            entry = RelocatedObject(
                original=self._objects[original_id],
                new_offset=new_offset,
            )
            relocations[original_id] = entry
            all_bytes += entry.original.get_bytes()

        all_bytes_mutable = bytearray(all_bytes)

        # (4) Within the set of reachable objects, rewrite the (now-relocated, but previously)
        # heap-allocated pointers to point amongst themselves.
        for relocated_object in relocations.values():
            if not relocated_object.original.is_pointer():
                continue

            # FIXME: We are not doing correct bits/bytes nomenclature here -- we need to make a
            # separate "Byte" and "Bit" wrapper type!!!
            byte_offset = relocated_object.new_offset.as_offset_in_bits()

            original_bytes = relocated_object.original.get_bytes()
            assert len(original_bytes) % _SIZE_BYTES_POINTER == 0

            relocated_bytes = b''
            for array_index in range(len(original_bytes) // _SIZE_BYTES_POINTER):
                cur_within_array_offset = _SIZE_BYTES_POINTER * array_index
                cur_pointer_bytes = original_bytes[cur_within_array_offset:(cur_within_array_offset + _SIZE_BYTES_POINTER)]

                (original_pointer_value,) = struct.unpack('P', cur_pointer_bytes)
                original_pointer_value = PythonCLevelPointerLocation(original_pointer_value)

                if relocated_pointer_target := relocations.get(original_pointer_value, None):
                    # FIXME: We are not doing correct bits/bytes nomenclature here -- we need to
                    # make a separate "Byte" and "Bit" wrapper type!!!
                    new_pointer_location = (
                        (self.MMAP_START_ADDRESS().as_offset_in_bits() +
                         (relocated_pointer_target.new_offset.as_offset_in_bits() *
                          _BITS_PER_BYTE)))
                    new_pointer_value = struct.pack('P', new_pointer_location)
                    assert len(new_pointer_value) == _SIZE_BYTES_POINTER

                    relocated_bytes += new_pointer_value

            all_bytes_mutable[byte_offset:(byte_offset + len(original_bytes))] = relocated_bytes

        return bytes(all_bytes_mutable)

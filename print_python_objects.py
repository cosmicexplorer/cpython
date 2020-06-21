import dataclasses
import glob
import itertools
import json
import logging
import os
import re
import sys
from collections import OrderedDict
from dataclasses import dataclass
from multiprocessing.pool import Pool
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import clang.cindex as clang


logger = logging.getLogger(__name__)


def _maybe_typedef_underlying(cursor: clang.Cursor) -> clang.Type:
    if cursor.kind == clang.CursorKind.TYPEDEF_DECL:
        return cursor.underlying_typedef_type
    return cursor.type


def _scrub_struct_prefix(s: str) -> str:
    return re.sub(r'^struct ', '', s)


def _get_non_pointer_type(ty: clang.Type) -> clang.Type:
    prev = ty
    cur = prev
    while not cur.kind == clang.TypeKind.INVALID:
        prev = cur
        cur = cur.get_pointee()
    return prev


@dataclass(frozen=True)
class StructFieldsEntry:
    decl: clang.Cursor
    fields: OrderedDict  # [str, Tuple[clang.Type, clang.Type]]

    def __post_init__(self) -> None:
        assert self.decl.kind == clang.CursorKind.STRUCT_DECL

    @property
    def type(self) -> clang.Type:
        return self.decl.type

    @property
    def record_size(self) -> int:
        return self.type.get_size()

    @property
    def type_name(self) -> str:
        return _scrub_struct_prefix(self.type.spelling)

    _pointer_types = [clang.TypeKind.POINTER]
    _array_types = [
        clang.TypeKind.CONSTANTARRAY,
        clang.TypeKind.INCOMPLETEARRAY,
        clang.TypeKind.VARIABLEARRAY,
        clang.TypeKind.DEPENDENTSIZEDARRAY,
    ]
    _func_types = [clang.TypeKind.FUNCTIONPROTO]

    @classmethod
    def _field_json(
            cls,
            field_name: str,
            clang_type: clang.Type,
            non_pointer_type: clang.Type,
            offset: int,
    ) -> Tuple[bool, Optional[int]]:
        is_pointer = False
        is_array = False
        array_size_hint: Optional[int] = None
        is_function_pointer = False

        if clang_type.kind in cls._pointer_types:
            is_pointer = True
        elif clang_type.kind in cls._array_types:
            is_pointer = True
            is_array = True
            array_size_hint = clang_type.get_array_size()

        non_pointer_decl = non_pointer_type.get_declaration()
        non_pointer_type = _maybe_typedef_underlying(non_pointer_decl)

        non_pointer_record_size = non_pointer_type.get_size()
        if non_pointer_record_size < 0:
            non_pointer_record_size = None
        non_pointer_type_name = _scrub_struct_prefix(non_pointer_type.spelling)

        if is_pointer:
            non_pointer_type = clang_type.get_pointee()

            if non_pointer_type.kind in cls._func_types:
                is_function_pointer = True

        type_name = _scrub_struct_prefix(clang_type.spelling)
        scrubbed_type_name = type_name
        if is_array:
            # An intrusive array will look like `setentry [8]` or `setentry []`. We erase that here
            # and pretend it's just a pointer. The `array_size_hint` will preserve that original
            # information in our output.
            scrubbed_type_name = re.sub(r'\[[0-9]*\].*$', '*', type_name)
            assert scrubbed_type_name != type_name

        if non_pointer_record_size is not None:
            # NB: convert bytes to bits!
            non_pointer_record_size = non_pointer_record_size * 8

        return dict(
            field_name=field_name,
            type_name=type_name,
            scrubbed_type_name=scrubbed_type_name,
            non_pointer_type_name=non_pointer_type_name,
            is_pointer=is_pointer,
            is_array=is_array,
            array_size_hint=array_size_hint,
            non_pointer_record_size=non_pointer_record_size,
            is_function_pointer=is_function_pointer,
            offset=offset,
        )

    def to_json(self) -> Dict[str, Any]:
        return dict(
            type_name=self.type_name,
            record_size=self.record_size,
            fields=[
                self._field_json(name,
                                 clang_type=clang_type,
                                 non_pointer_type=non_pointer_type,
                                 offset=self.type.get_offset(name))
                for name, (clang_type, non_pointer_type) in self.fields.items()
            ],
        )


@dataclass(frozen=True)
class PyObjectEntry:
    python_level_name: str
    py_object_struct_name: str

    def to_json(self, *, scrub_struct_prefix=True) -> Dict[str, str]:
        return dict(
            python_level_name=self.python_level_name,
            py_object_struct_name=(
                _scrub_struct_prefix(self.py_object_struct_name)
                if scrub_struct_prefix else
                self.py_object_struct_name
            ),
        )


@dataclass(frozen=True)
class CPythonKnowledge:
    ob_type_offset: Optional[int]
    tp_name_offset: Optional[int]
    struct_field_mapping: List[StructFieldsEntry]
    py_object_mapping: List[PyObjectEntry]

    @classmethod
    def _merge_optional_equal_values(cls, a, b):
        if a is not None:
            if b is not None:
                assert a == b
            return a
        return b

    def merge(self, other: 'CPythonKnowledge') -> 'CPythonKnowledge':
        return dataclasses.replace(
            self,
            ob_type_offset=self._merge_optional_equal_values(self.ob_type_offset,
                                                             other.ob_type_offset),
            tp_name_offset=self._merge_optional_equal_values(self.tp_name_offset,
                                                               other.tp_name_offset),
            struct_field_mapping=(self.struct_field_mapping + other.struct_field_mapping),
            py_object_mapping=(self.py_object_mapping + other.py_object_mapping),
        )

    def to_json(self) -> Dict[str, Any]:
        return dict(
            ob_type_offset=self.ob_type_offset,
            tp_name_offset=self.tp_name_offset,
            struct_field_mapping={
                # NB: We "normalize" types here by removing the "struct" keyword in front. Why?
                # Because for some structs such as `zipobject` (which is only ever defined as
                # `typedef struct { ... } zipobject;`), the `.py_object_mapping` below will have a
                # `struct ` in there that doesn't seem to exist in any real source file.
                # Pretty weird!!!!
                entry.type_name: entry.to_json()
                for entry in self.struct_field_mapping
                if entry.record_size > 0
            },
            py_object_mapping={
                # NB: We also "normalize" by scrubbing any `struct ` prefixes in the
                # `.py_object_mapping` values!
                entry.python_level_name: entry.to_json(scrub_struct_prefix=True)
                for entry in self.py_object_mapping
            },
        )


@dataclass(frozen=True)
class FileToParse:
    path: Path

    def parse(self, index: clang.Index) -> CPythonKnowledge:
        python_tu = index.parse(
            str(self.path),
            args=['-x', 'c', '-I.', '-IInclude', '-IObjects', '-IPython', '-IDoc/includes'])

        ob_type_offset: Optional[int] = None
        tp_name_offset: Optional[int] = None

        struct_field_mapping: List[StructFieldsEntry] = []
        py_object_mapping: List[PyObjectEntry] = []

        for cursor in python_tu.cursor.walk_preorder():
            try:
                kind = cursor.kind
            except ValueError:
                continue

            if kind == clang.CursorKind.TYPEDEF_DECL:
                typedef_type = cursor.type
                underlying_type = cursor.underlying_typedef_type

                if typedef_type.spelling == 'PyObject':
                    assert ob_type_offset is None
                    if underlying_type.spelling != 'struct _object':
                        continue
                    ob_type_offset = underlying_type.get_offset('ob_type')
                elif typedef_type.spelling == 'PyTypeObject':
                    assert tp_name_offset is None
                    if underlying_type.spelling != 'struct _typeobject':
                        continue
                    tp_name_offset = underlying_type.get_offset('tp_name')

            # Traverse struct declarations to find ones for each registered PyObject "subclass".
            elif kind == clang.CursorKind.STRUCT_DECL:
                struct_type = cursor.type

                # if struct_type.spelling == 'PySetObject':
                #     import pdb; pdb.set_trace()

                fields: List[Tuple[str, Tuple[clang.Type, clang.Type]]] = []
                for field in struct_type.get_fields():
                    non_pointer_target_type = _get_non_pointer_type(field.type)

                    type_decl_for_field = field.type.get_declaration()
                    if type_decl_for_field.kind == clang.CursorKind.TYPEDEF_DECL:
                        field_target_type = type_decl_for_field.underlying_typedef_type
                    else:
                        field_target_type = field.type

                    fields.append((field.spelling, (field_target_type, non_pointer_target_type)))

                    # Check for PyObject_HEAD.
                    # NB: We intentionally access `field.type`, NOT `field_target_type`, because we
                    # want to check whether the typedef'd named is PyObject, not the source struct!
                    if field.type.spelling == 'PyObject':
                        assert field.spelling == 'ob_base'
                        # This is always the first element of any PyObject struct.
                        assert struct_type.get_offset(field.spelling) == 0

                entry = StructFieldsEntry(
                    decl=cursor,
                    fields=OrderedDict(fields),
                )
                struct_field_mapping.append(entry)

            # Find PyTypeObject usages in order to map PyObject subclasses to the names of the
            # builtin types that represent them!
            elif kind == clang.CursorKind.VAR_DECL:
                if not cursor.is_definition():
                    continue
                if cursor.type.spelling == 'PyTypeObject':
                    name_literal = None
                    basicsize_sizeof_object = None

                    try:
                        decl_args = list(list(cursor.get_children())[-1].get_children())
                    except Exception as e:
                        logger.warning(str([[tok.spelling for tok in cur.get_tokens()]
                                            for cur in cursor.get_children()]))
                        logger.exception(e)
                        continue

                    name_args = tuple(decl_args[1].get_tokens())
                    if not name_args:
                        continue
                    name_literal = name_args[-1].spelling.replace('"', '')

                    try:
                        object_layout_target = list(
                            list(decl_args[2].get_children())[-1]
                            .get_children())[0]
                    except Exception as e:
                        logger.warning(str([[tok.spelling for tok in cur.get_tokens()]
                                            for cur in decl_args[2].get_children()]))
                        logger.exception(e)
                        continue
                    type_defn = object_layout_target.get_definition()
                    if type_defn is not None:
                        basicsize_sizeof_object = _maybe_typedef_underlying(type_defn)

                    if not (name_literal and basicsize_sizeof_object):
                        continue
                    entry = PyObjectEntry(
                        python_level_name=name_literal,
                        py_object_struct_name=basicsize_sizeof_object.spelling,
                    )
                    py_object_mapping.append(entry)

        return CPythonKnowledge(
            ob_type_offset=ob_type_offset,
            tp_name_offset=tp_name_offset,
            struct_field_mapping=struct_field_mapping,
            py_object_mapping=py_object_mapping,
        )


# FIXME: local hack!!!
if os.uname().sysname == 'Darwin':
    clang.Config.set_library_file('/usr/local/opt/llvm/lib/libclang.dylib')
else:
    clang.Config.set_library_file('/usr/lib/llvm-6.0/lib/libclang.so.1')

index = clang.Index.create()


cpython_accumulated_knowledge = FileToParse(Path('Include/Python.h')).parse(index)

# cpython_accumulated_knowledge = cpython_accumulated_knowledge.merge(FileToParse(Path('Include/cpython/object.h')).parse(index))

all_files_to_parse = [
    FileToParse(Path(src))
    for src in itertools.chain(*[
            glob.glob(pat)
            for d in ['Objects', 'Modules', 'Python']
            # NB: Why isn't this the same as **/*.c???????????
            for pat in [f'{d}/*.c', f'{d}/**/*.c']
    ])
    # for src in itertools.chain(glob.glob('**/*.c'),
    #                            glob.glob('**/*.h'))
]

def parse_c_source_file(to_parse: FileToParse) -> CPythonKnowledge:
    try:
        return to_parse.parse(index)
    except Exception as e:
        raise Exception(f'failed in file {to_parse}: {e}') from e


# NB: Due to extremely coarse locking in the libclang API, splitting this across threads actually
# *hurts* performance.
# TODO: try subprocesses?
for to_parse in all_files_to_parse:
    parsed = parse_c_source_file(to_parse)
    cpython_accumulated_knowledge = cpython_accumulated_knowledge.merge(parsed)


sys.stdout.write(json.dumps(cpython_accumulated_knowledge.to_json(), indent=4) + '\n')

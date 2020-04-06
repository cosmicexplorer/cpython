import dataclasses
import glob
import json
import re
import sys
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import clang.cindex as clang


def _maybe_typedef_underlying(cursor: clang.Cursor) -> clang.Type:
    if cursor.kind == clang.CursorKind.TYPEDEF_DECL:
        return cursor.underlying_typedef_type
    return cursor.type


@dataclass(frozen=True)
class StructFieldsEntry:
    decl: clang.Cursor
    fields: OrderedDict  # [str, clang.Type]

    def __post_init__(self) -> None:
        assert self.decl.kind == clang.CursorKind.STRUCT_DECL

    @property
    def type(self) -> clang.Type:
        return self.decl.type

    @property
    def type_name(self) -> str:
        return self.type.spelling

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

        non_pointer_type = clang_type
        non_pointer_record_size: Optional[int] = None
        non_pointer_type_name: Optional[str] = None
        if is_pointer:
            non_pointer_type = clang_type.get_pointee()

            if non_pointer_type.kind in cls._func_types:
                is_function_pointer = True
        non_pointer_decl = non_pointer_type.get_declaration()
        non_pointer_type = _maybe_typedef_underlying(non_pointer_decl)

        if (not is_function_pointer) and (not is_array):
            non_pointer_record_size = non_pointer_type.get_size()
            non_pointer_type_name = non_pointer_type.spelling

        type_name = clang_type.spelling
        scrubbed_type_name = type_name
        if is_array:
             scrubbed_type_name = re.sub(r'\[[0-9]*\].*$', '*', type_name)
             assert scrubbed_type_name != type_name

        return dict(
            field_name=field_name,
            type_name=type_name,
            scrubbed_type_name=scrubbed_type_name,
            is_pointer=is_pointer,
            is_array=is_array,
            array_size_hint=array_size_hint,
            non_pointer_type_name=non_pointer_type_name,
            non_pointer_record_size=non_pointer_record_size,
            is_function_pointer=is_function_pointer,
            offset=offset,
        )

    def to_json(self) -> Dict[str, Any]:
        return dict(
            type_name=self.type_name,
            fields=[
                self._field_json(name, clang_type, offset=self.type.get_offset(name))
                for name, clang_type in self.fields.items()
            ],
        )


@dataclass(frozen=True)
class PyObjectEntry:
    python_level_name: str
    py_object_struct_name: str

    def to_json(self) -> Dict[str, str]:
        return dict(
            python_level_name=self.python_level_name,
            py_object_struct_name=self.py_object_struct_name,
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
                entry.type_name: entry.to_json() for entry in self.struct_field_mapping
            },
            py_object_mapping={
                entry.python_level_name: entry.to_json() for entry in self.py_object_mapping
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
                    assert underlying_type.spelling == 'struct _object'
                    ob_type_offset = underlying_type.get_offset('ob_type')
                elif typedef_type.spelling == 'PyTypeObject':
                    assert tp_name_offset is None
                    assert underlying_type.spelling == 'struct _typeobject'
                    tp_name_offset = underlying_type.get_offset('tp_name')

            # Traverse struct declarations to find ones for each registered PyObject "subclass".
            elif kind == clang.CursorKind.STRUCT_DECL:
                struct_type = cursor.type

                # if struct_type.spelling == 'PySetObject':
                #     import pdb; pdb.set_trace()

                fields: List[Tuple[str, clang.Type]] = []
                for field in struct_type.get_fields():
                    field_target_type = field.type
                    type_decl_for_field = field_target_type.get_declaration()
                    if type_decl_for_field.kind == clang.CursorKind.TYPEDEF_DECL:
                        field_target_type = type_decl_for_field.underlying_typedef_type

                    fields.append((field.spelling, field_target_type))

                    # Check for PyObject_HEAD.
                    # NB: We intentionally access `field.type`, NOT `field_target_type`, because we want to
                    # check whether the typedef'd named is PyObject, not the source struct!
                    if field.type.spelling == 'PyObject':
                        assert field.spelling == 'ob_base'
                        # This is always the first element of any PyObject struct.
                        assert struct_type.get_offset(field.spelling) == 0

                entry = StructFieldsEntry(
                    decl=cursor,
                    fields=OrderedDict(fields),
                )
                struct_field_mapping.append(entry)

            # Find PyTypeObject usages in order to map PyObject subclasses to the names of the builtin types
            # that represent them!
            elif kind == clang.CursorKind.VAR_DECL:
                if not cursor.is_definition():
                    continue
                if cursor.type.spelling == 'PyTypeObject':
                    name_literal = None
                    basicsize_sizeof_object = None

                    for c in cursor.walk_preorder():
                        if c.kind == clang.CursorKind.STRING_LITERAL:
                            if name_literal is not None:
                                continue
                            name_literal = c.spelling.replace('"', '')
                        elif c.kind == clang.CursorKind.CXX_UNARY_EXPR:
                            if basicsize_sizeof_object is not None:
                                continue
                            args = list(c.get_children())
                            if len(args) != 1:
                                continue
                            # assert len(args) == 1, f'len(args): {len(args)} was not 1!'
                            type_defn = args[0].get_definition()
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


clang.Config.set_library_file('/usr/local/opt/llvm/lib/libclang.dylib')


index = clang.Index.create()


cpython_accumulated_knowledge = FileToParse(Path('Include/Python.h')).parse(index)

for d in ['Objects', 'Python', 'Modules']:
    for src in glob.glob(f'{d}/*.c'):
        to_parse = FileToParse(Path(src))
        try:
            parsed = to_parse.parse(index)
            cpython_accumulated_knowledge = cpython_accumulated_knowledge.merge(parsed)
        except Exception as e:
            raise Exception(f'failed in file {src}: {e}') from e

sys.stdout.write(json.dumps(cpython_accumulated_knowledge.to_json()) + '\n')

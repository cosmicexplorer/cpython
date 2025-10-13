from __future__ import annotations

import abc
import code
from dataclasses import dataclass, field
from importlib.abc import Loader
import importlib.util
import os
import re
import sys
from collections.abc import Iterator, Iterable
from contextlib import contextmanager
from importlib.machinery import ModuleSpec
from types import ModuleType, TracebackType
from typing import Any, ClassVar, Generic, Protocol, TypeVar, cast
from typing_extensions import Self

class VersionModuleError(Exception):
    """Parent class for much more specific exceptions."""

class CLIParseError(VersionModuleError):
    def __init__(self, argv: list[str]) -> None:
        super().__init__(
            "error parsing cli args:\n"
            f"{argv[0]} module-name [path-entry...]\n"
            f"got: {argv!r}"
        )
        self.argv = argv


class VersionParseError(VersionModuleError):
    def __init__(self, s: str, rx: re.Pattern[str]) -> None:
        super().__init__(
            f"simple version parsing failed: {s!r} did not match desired pattern {rx!r}"
        )
        self.s = s
        self.rx = rx

@dataclass(frozen=True, slots=True, order=True)
class SimpleVersion:
    """It would be nice to vendor the `packaging` library into the stdlib somehow.

    Due to `ensurepip`, it technically already is.

    Since this class is specifically intended for parsing pip version strings, which conform to
    these simplifying assumptions.
    """
    major: int
    minor: int | None
    patch: int | None

    _digit_rx: ClassVar[re.Pattern[str]] = re.compile(r'([0-9]+)(?:\.([0-9]+))?(?:\.([0-9]+))?')

    @classmethod
    def parse(cls, s: str) -> Self:
        m = cls._digit_rx.match(s)
        if m is None:
            raise VersionParseError(s, cls._digit_rx)
        (major, minor, patch) = m.groups()
        return cls(int(major),
                   int(minor) if minor is not None else None,
                   int(patch) if patch is not None else None)

    def __iter__(self) -> Iterator[int]:
        yield self.major
        if self.minor is None:
            return
        yield self.minor
        if self.patch is None:
            return
        yield self.patch

    def __str__(self) -> str:
        return '.'.join(map(str, self))

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}.parse({self})"


try:
    (source, module_name, requisite_version, *overwrite_path) = sys.argv
    version_to_match = SimpleVersion.parse(requisite_version)
except ValueError as e:
    raise CLIParseError(sys.argv) from e


class IndirectAccessor[O](abc.ABC):
    @abc.abstractmethod
    def set_global_state(self, val: O) -> None: ...
    @abc.abstractmethod
    def get_global_state(self) -> O: ...


@dataclass(frozen=True, slots=True)
class SysPathAccessor(IndirectAccessor[list[str]]):
    sys_mod: ModuleType

    @classmethod
    def create(cls, *, sys_mod: ModuleType | None = None) -> Self:
        if sys_mod is None:
            sys_mod = sys
        return cls(sys_mod)

    def set_global_state(self, val: list[str]) -> None:
        if not val:
            return
        self.sys_mod.path[:] = val

    def get_global_state(self) -> list[str]:
        return cast(list[str], self.sys_mod.path)[:]


class StackAccessor[O](abc.ABC):
    @abc.abstractmethod
    def push(self, val: O) -> None: ...
    @abc.abstractmethod
    def pop(self) -> O: ...


class StateManagementError(VersionModuleError):
    def __init__(self, current_value: Any, error_description: str) -> None:
        super().__init__(
            f"an assertion was broken: {error_description}\n"
            f"the state at the time was: {current_value}"
        )
        self.current_value = current_value

@dataclass(frozen=True, unsafe_hash=False, slots=True)
class ListAccessor[O, I](StackAccessor[O]):
    source: I
    handles: list[O] = field(default_factory=list)

    @classmethod
    def create(
        cls: type[ListAccessor[list[str], SysPathAccessor]],
        *,
        source: SysPathAccessor | None = None,
    ) -> ListAccessor[list[str], SysPathAccessor]:
        if source is None:
            source = SysPathAccessor.create()
        return cls(source)

    def push[I2: IndirectAccessor[O]](self: ListAccessor[O, I2], val: O) -> None:
        self.handles.append(self.source.get_global_state())
        self.source.set_global_state(val)

    def pop[I2: IndirectAccessor[O]](self: ListAccessor[O, I2]) -> O:
        ret = self.source.get_global_state()
        if not self.handles:
            raise StateManagementError(ret, "no stack entries upon receiving pop request")
        self.source.set_global_state(self.handles.pop())
        return ret


@dataclass(frozen=True, unsafe_hash=False, slots=True)
class RewriteSysPath:
    _stack: ListAccessor[list[str], SysPathAccessor]
    mutable_paths: list[str] = field(default_factory=list)

    @classmethod
    def create(
        cls,
        *,
        stack: ListAccessor[list[str], SysPathAccessor] | None = None,
    ) -> Self:
        if stack is None:
            stack = ListAccessor[list[str], SysPathAccessor].create()
        return cls(stack)

    def __enter__(self) -> Self:
        # NB: does *not* clear out or move the existing paths!
        self._stack.push(self.mutable_paths[:])
        return self

    def __exit__(
        self,
        exc_ty: type[BaseException] | None,
        exc_val: BaseException | None,
        tb: TracebackType | None,
    ) -> bool | None:
        # Back to the state we were before pushing!
        self.mutable_paths[:] = self._stack.pop()
        return None


class ModuleSearchFailure(VersionModuleError):
    def __init__(self, module_name: str, path: list[str]) -> None:
        super().__init__(f"could not find {module_name!r} in {path!r}")
        self.module_name = module_name
        self.path = path


class ExecLoader(Loader):
    """There is a specific note in the stdlib that removing `exec_module()` is for `hasattr()`[]
    backwards compatibility reasons (9e09849d20987c131b28bcdd252e53440d4cd1b3).
    So that's why this is here."""
    def exec_module(self, module: ModuleType) -> None:
        raise NotImplementedError


class ExecModule(ModuleType):
    __loader__: ExecLoader


@dataclass(slots=True)
class SpecSearcher:
    spec: ModuleSpec
    sys_mod: ModuleType
    mod: ExecModule | None = None

    @classmethod
    def search_in_current_env(
        cls,
        module_name: str,
        *,
        sys_mod: ModuleType | None = None,
    ) -> Self | None:
        # This identifies whether the module is importable, entirely separate from executing it.
        spec = importlib.util.find_spec(module_name)
        if spec is None:
            return None
        if sys_mod is None:
            sys_mod = sys
        return cls(spec, sys_mod)

    def get_or_load_module(self) -> ModuleType:
        # Very basic caching is ok since modules are very very large objects.
        if self.mod is not None:
            return self.mod
        # This calls create_module() in a way I can't figure out how to get working myself.
        mod = cast(ExecModule, importlib.util.module_from_spec(self.spec))
        # Add it to the known modules (this forms  strong reference).
        self.sys_mod.modules[mod.__name__] = mod
        # Finally, execute its code (this is the recommended order of operations).
        mod.__loader__.exec_module(mod)
        self.mod = mod
        return self.mod

    def __enter__(self) -> ModuleType:
        if self.mod is not None:
            raise StateManagementError(
                self.mod,
                f"should not have attempted to load the same module twice",
            )
        return self.get_or_load_module()

    def drop_module(self) -> None:
        # Very basic caching is ok here since modules are large.
        if self.mod is None:
            return
        mod = self.sys_mod.modules.pop(self.mod.__name__, None)
        if mod is None:
            raise StateManagementError(mod, "module popped off sys modules was None")
        if mod is not self.mod:
            raise StateManagementError(
                mod,
                "the module popped off sys modules was not the same allocation "
                f"as the {self.mod} handle"
            )
        self.mod = None

    def __exit__(
        self,
        exc_ty: type[BaseException] | None,
        exc_val: BaseException | None,
        tb: TracebackType | None,
    ) -> bool | None:
        if self.mod is None:
            raise StateManagementError(
                self.mod,
                f"should have executed module for spec {self.spec} and retained a handle to it",
            )
        self.drop_module()
        return None


class VersionCompatError(VersionModuleError):
    def __init__(self, resource: str, requested: SimpleVersion, found: SimpleVersion, desc: str) -> None:
        super().__init__(
            f"bootstrap resource {resource!r} had a version incompatibility: {desc}\n"
            f"requested: {requested}\n"
            f"found: {found}"
        )
        self.requested = requested
        self.found = found


@dataclass(frozen=True, slots=True, unsafe_hash=False)
class ModuleSearcher:
    _state: RewriteSysPath

    @classmethod
    def create(cls, state: RewriteSysPath | None = None) -> Self:
        if state is None:
            state = RewriteSysPath.create()
        return cls(state)

    @contextmanager
    def search_for_module(
        self,
        module_name: str,
        path: Iterable[str] | None = None,
    ) -> Iterator[ModuleType]:
        if path:
            self._state.mutable_paths[:] = list(path)
        with self._state:
            # See if the module is importable at all first.
            spec_searcher = SpecSearcher.search_in_current_env(module_name)
            if spec_searcher is None:
                raise ModuleSearchFailure(module_name, list(path or []))

            # Now load the module!
            with spec_searcher as mod:
                # This is also a contextmanager so it can drop the module afterwards.
                yield mod




module_searcher = ModuleSearcher.create()

with module_searcher.search_for_module(
    module_name,
    path=overwrite_path,
) as mod:
    v = SimpleVersion.parse(mod.__version__)
    if v < version_to_match:
        raise VersionCompatError(mod.__name__, version_to_match, v, "version was too low")
    print(str(v))

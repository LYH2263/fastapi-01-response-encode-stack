"""Request scoped response serialization policy stack.

The ``response_model_*`` arguments of path operation decorators (``exclude_none``,
``by_alias``, ``exclude_unset``, ``exclude_defaults``) are fixed for the whole
response. This module provides a stack that allows dependencies, middleware and
path operation code to tighten or relax those switches for a nested section of a
single request, and to restore the previous state automatically.

The stack is independent from:

* the path operation decorator defaults (they are used as the *baseline* that
  stack frames are merged on top of), and
* ``app.dependency_overrides`` (overriding dependencies never touches this
  stack and vice versa).

Concurrency
-----------

The stack hangs on a module level :class:`~contextvars.ContextVar`, i.e. a
single shared mount point that every concurrent request enters. Each request
(HTTP request, or a WebSocket connection after the handshake) :dfn:`binds` its
own, private, request-owned frame list on that mount point for the lifetime of
the request. Because ASGI transports run every request in its own task (and
``ContextVar`` values are copied when a task or worker thread is spawned), two
concurrent requests pushing different switches never share a frame, while
nested code inside the same request (dependencies running in worker threads,
stream producer tasks, ...) reaches the same request-owned stack.

Frames pushed but never popped die together with the request-owned list when
the request ends and the bind is reset; they can never leak into a later
request.
"""

from __future__ import annotations

import contextlib
import warnings
from collections.abc import Iterator
from contextvars import ContextVar, Token
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Literal, TypedDict

if TYPE_CHECKING:  # pragma: no cover
    from fastapi._compat import ModelField

__all__ = [
    "ValidationMode",
    "ResponseSerializationPolicy",
    "ResolvedResponseSerializationPolicy",
    "ResponsePolicyPushToken",
    "push_response_policy",
    "pop_response_policy",
    "response_policy",
    "get_pushed_response_policy",
    "resolve_response_policy",
    "bind_response_policy_stack",
    "response_policy_stack_empty",
]


class ValidationMode(str, Enum):
    """How response model validation errors are handled for one stack layer.

    * ``strict`` (the default): response model errors keep raising
      :class:`fastapi.exceptions.ResponseValidationError`, exactly like today.
    * ``lenient``: the response model validation pass is skipped; the payload is
      still serialized honoring the other switches (``exclude_none``,
      ``by_alias``, ...).
    """

    strict = "strict"
    lenient = "lenient"


ValidationModeValue = ValidationMode | Literal["strict", "lenient"]

_POLICY_FIELDS = (
    "exclude_none",
    "by_alias",
    "exclude_unset",
    "exclude_defaults",
    "validation_mode",
)


@dataclass(frozen=True)
class ResponseSerializationPolicy:
    """One frame of the response serialization policy stack.

    Every field is optional: a ``None`` field means "inherit from the frame
    below (or from the route baseline)". A frame that only sets
    ``exclude_none=True`` therefore only tightens that single switch.
    """

    exclude_none: bool | None = None
    by_alias: bool | None = None
    exclude_unset: bool | None = None
    exclude_defaults: bool | None = None
    validation_mode: ValidationModeValue | None = None


@dataclass(frozen=True)
class ResolvedResponseSerializationPolicy:
    """The concrete, merged view used by a serialization exit point.

    All values are booleans/enums: every inheritance has already been resolved
    against the outer frames and the route baseline.
    """

    exclude_none: bool
    by_alias: bool
    exclude_unset: bool
    exclude_defaults: bool
    validation_mode: ValidationMode

    @property
    def dump_kwargs(self) -> DumpKwargs:
        """The four encoder switches, ready to forward to a serializer."""
        return {
            "by_alias": self.by_alias,
            "exclude_unset": self.exclude_unset,
            "exclude_defaults": self.exclude_defaults,
            "exclude_none": self.exclude_none,
        }


class DumpKwargs(TypedDict):
    """The four serialization switches accepted by the serializers."""

    by_alias: bool
    exclude_unset: bool
    exclude_defaults: bool
    exclude_none: bool


class ResponsePolicyPushToken:
    """Opaque token returned by :func:`push_response_policy`.

    It exists so that :func:`pop_response_policy` can verify that pushes and
    pops are balanced and correctly nested. It is not meant to be inspected.
    """

    __slots__ = ("frame",)

    def __init__(self, frame: ResponseSerializationPolicy) -> None:
        self.frame = frame


#: Shared mount point. ``None`` means "no stack bound in this context yet".
#: Every request binds its own private frame list for its lifetime; the var
#: itself is a single, process-wide, concurrency-safe entry point.
_response_policy_stack_var: ContextVar[list[ResponseSerializationPolicy] | None] = (
    ContextVar("fastapi_response_policy_stack", default=None)
)


def _coerce_validation_mode(
    value: ValidationModeValue | None,
) -> ValidationMode | None:
    if value is None:
        return None
    if isinstance(value, ValidationMode):
        return value
    if isinstance(value, str):
        try:
            return ValidationMode(value)
        except ValueError:
            pass
    raise ValueError(
        "validation_mode must be ValidationMode.strict, "
        f"ValidationMode.lenient, 'strict' or 'lenient', got: {value!r}"
    )


def _build_frame(
    policy: ResponseSerializationPolicy | None,
    overrides: dict[str, Any],
) -> ResponseSerializationPolicy:
    if policy is not None and not isinstance(policy, ResponseSerializationPolicy):
        raise TypeError(
            "policy must be a ResponseSerializationPolicy instance, "
            f"got: {type(policy)!r}"
        )
    values: dict[str, Any] = {
        name: getattr(policy, name) if policy is not None else None
        for name in _POLICY_FIELDS
    }
    for key, value in overrides.items():
        if key not in _POLICY_FIELDS:
            raise TypeError(f"Unknown response serialization policy field: {key!r}")
        # A None keyword means "not set": it must not clobber a field that
        # was explicitly provided through a ResponseSerializationPolicy.
        if value is not None:
            values[key] = value
    values["validation_mode"] = _coerce_validation_mode(values["validation_mode"])
    return ResponseSerializationPolicy(**values)


def push_response_policy(
    policy: ResponseSerializationPolicy | None = None,
    *,
    exclude_none: bool | None = None,
    by_alias: bool | None = None,
    exclude_unset: bool | None = None,
    exclude_defaults: bool | None = None,
    validation_mode: ValidationModeValue | None = None,
) -> ResponsePolicyPushToken:
    """Push one frame on top of the current request's policy stack.

    The frame only overrides the fields explicitly passed; every other field
    keeps inheriting from outer frames and the route baseline. Pair the call
    with :func:`pop_response_policy`, or prefer the :func:`response_policy`
    context manager which pops automatically, including on exception paths.

    Calling this outside of a request binds a stack to the current context on
    first use; framework request handlers always bind their own fresh stack,
    so such a frame can never leak into a request.
    """
    frame = _build_frame(
        policy,
        {
            "exclude_none": exclude_none,
            "by_alias": by_alias,
            "exclude_unset": exclude_unset,
            "exclude_defaults": exclude_defaults,
            "validation_mode": validation_mode,
        },
    )
    stack = _response_policy_stack_var.get()
    if stack is None:
        stack = []
        _response_policy_stack_var.set(stack)
    stack.append(frame)
    return ResponsePolicyPushToken(frame)


def pop_response_policy(token: ResponsePolicyPushToken) -> None:
    """Pop the frame pushed by the matching :func:`push_response_policy`.

    Raises ``RuntimeError`` if the stack is empty or if the frame on top does
    not belong to ``token`` (unbalanced / cross nesting between push and pop).
    """
    if not isinstance(token, ResponsePolicyPushToken):
        raise TypeError(
            "pop_response_policy() expects the token returned by push_response_policy()"
        )
    stack = _response_policy_stack_var.get()
    if not stack:
        raise RuntimeError(
            "Cannot pop a response serialization policy frame: the current "
            "policy stack is empty"
        )
    if stack[-1] is not token.frame:
        raise RuntimeError(
            "Cannot pop a response serialization policy frame: another frame "
            "has been pushed on top of it, pushes and pops must be properly "
            "nested"
        )
    stack.pop()


@contextlib.contextmanager
def response_policy(
    policy: ResponseSerializationPolicy | None = None,
    *,
    exclude_none: bool | None = None,
    by_alias: bool | None = None,
    exclude_unset: bool | None = None,
    exclude_defaults: bool | None = None,
    validation_mode: ValidationModeValue | None = None,
) -> Iterator[ResponsePolicyPushToken]:
    """Context manager pushing a frame for its body and popping it on exit.

    The frame is always popped, including when the body exits through an
    exception, so an inner section can never forget to restore the outer user
    policy::

        with response_policy(exclude_none=True):
            ...  # tight "drop None" section

    Can be used from ``async def`` code, dependencies, middleware and
    generator based streaming endpoints alike.
    """
    token = push_response_policy(
        policy,
        exclude_none=exclude_none,
        by_alias=by_alias,
        exclude_unset=exclude_unset,
        exclude_defaults=exclude_defaults,
        validation_mode=validation_mode,
    )
    try:
        yield token
    finally:
        pop_response_policy(token)


def _fold_frames(
    frames: list[ResponseSerializationPolicy],
) -> ResponseSerializationPolicy | None:
    merged: dict[str, Any] = dict.fromkeys(_POLICY_FIELDS)
    for frame in frames:
        for name in _POLICY_FIELDS:
            value = getattr(frame, name)
            if value is not None:
                merged[name] = value
    if all(merged[name] is None for name in _POLICY_FIELDS):
        return None
    return ResponseSerializationPolicy(**merged)


def get_pushed_response_policy() -> ResponseSerializationPolicy | None:
    """Folded view of the user frames currently on the stack.

    The returned policy only contains fields set by at least one frame; unset
    fields stay ``None`` (to be inherited from the route baseline). Returns
    ``None`` when the current request never pushed anything, which is exactly
    the state in which every exit point must behave like the decorator-only
    world.
    """
    stack = _response_policy_stack_var.get()
    if not stack:
        return None
    return _fold_frames(stack)


def resolve_response_policy(
    *,
    by_alias: bool = True,
    exclude_unset: bool = False,
    exclude_defaults: bool = False,
    exclude_none: bool = False,
    validation_mode: ValidationModeValue = ValidationMode.strict,
) -> ResolvedResponseSerializationPolicy:
    """Merge the route baseline with the current request's folded stack top.

    The keyword arguments are the *route baseline* (the resolved
    ``response_model_*`` decorator values for the current exit). Stack frames
    override every field they set; fields no frame sets keep the baseline.
    This only resolves switches; every serialization exit point keeps its own
    validation/encoding/wire code.
    """
    folded = get_pushed_response_policy()

    def resolve(name: str, baseline: Any) -> Any:
        if folded is not None:
            override = getattr(folded, name)
            if override is not None:
                return override
        return baseline

    return ResolvedResponseSerializationPolicy(
        exclude_none=bool(resolve("exclude_none", exclude_none)),
        by_alias=bool(resolve("by_alias", by_alias)),
        exclude_unset=bool(resolve("exclude_unset", exclude_unset)),
        exclude_defaults=bool(resolve("exclude_defaults", exclude_defaults)),
        validation_mode=_coerce_validation_mode(
            resolve("validation_mode", validation_mode)
        )
        or ValidationMode.strict,
    )


@contextlib.contextmanager
def bind_response_policy_stack() -> Iterator[None]:
    """Bind a fresh, request owned, empty policy stack for the current context.

    Frames an outer middleware already pushed in an inherited context are
    copied into the fresh stack so they keep applying; mutations inside the
    request stay on the request-owned copy and never reach the outer stack.
    The bind is always reset on exit, so frames leaked by user code cannot
    survive the request.
    """
    inherited = _response_policy_stack_var.get()
    token: Token[list[ResponseSerializationPolicy] | None] = (
        _response_policy_stack_var.set(list(inherited) if inherited else [])
    )
    try:
        yield
    finally:
        _response_policy_stack_var.reset(token)


def best_effort_construct(field: ModelField, value: Any) -> Any:
    """Coerce a raw dict into the field's model without validating it.

    Lenient mode must skip response model validation, yet the model-level
    serialization switches (``by_alias``, ``exclude_none`` ...) only take
    effect when pydantic serializes a model instance: serializing a plain dict
    through a model's ``TypeAdapter`` treats it as an opaque dict and the
    switches are ignored. Building the instance with ``model_construct`` is
    non-validating by definition, so ill-typed values still flow through while
    serialization keeps honoring the stack switches. Only the outermost dict
    is converted; nested values remain exactly as provided.
    """
    if not isinstance(value, dict):
        return value
    type_adapter = getattr(field, "_type_adapter", None)
    if type_adapter is None:
        return value
    from pydantic import BaseModel

    model_cls: type[BaseModel] | None = None
    stack: list[Any] = [getattr(type_adapter, "core_schema", None)]
    seen: set[int] = set()
    while stack:
        schema = stack.pop()
        if not isinstance(schema, dict):
            continue
        marker = id(schema)
        if marker in seen:
            continue
        seen.add(marker)
        cls = schema.get("cls")
        if (
            schema.get("type") == "model"
            and isinstance(cls, type)
            and issubclass(cls, BaseModel)
        ):
            model_cls = cls
            break
        stack.extend(schema.values())
    if model_cls is None:
        return value
    try:
        return model_cls.model_construct(**value)
    except Exception:
        return value


@contextlib.contextmanager
def lenient_serialization() -> Iterator[None]:
    """Silence best-effort serializer warnings in a lenient validation section.

    Lenient mode skips response model validation and hands the raw payload to
    the serializer, which then emits best-effort ``UserWarning`` messages for
    values that do not match the declared type. Lenient mode is opt-in and is
    expected to return those payloads, so the warnings must not turn into
    errors.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"Pydantic serializer warnings.*",
            category=UserWarning,
        )
        yield


@contextlib.contextmanager
def response_policy_stack_empty() -> Iterator[None]:
    """Run a section with an empty policy stack view.

    Used for :class:`fastapi.BackgroundTasks`, which run after the response
    has been handed to the client: they must observe an empty stack, even when
    the request that scheduled them pushed frames (and even when one of those
    frames was never popped). Frames pushed inside the section itself are
    discarded on exit.
    """
    token: Token[list[ResponseSerializationPolicy] | None] = (
        _response_policy_stack_var.set([])
    )
    try:
        yield
    finally:
        _response_policy_stack_var.reset(token)

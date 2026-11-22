"""Typed planner configuration: each planner declares a dataclass of its
options, and user overrides are checked against it before anything runs."""

from __future__ import annotations

import dataclasses
import types
import typing
from typing import Any, Mapping

_NONE_WORDS = ("none", "null")
_BOOL_WORDS = {"true": True, "yes": True, "1": True, "false": False, "no": False, "0": False}


def _hint_members(hint) -> tuple:
    origin = typing.get_origin(hint)
    if origin is typing.Union or origin is types.UnionType:
        return typing.get_args(hint)
    return (hint,)


def _coerce(name: str, value: Any, hint) -> Any:
    """Convert a string from the command line or a TOML file to the field's type."""
    if not isinstance(value, str):
        return value
    members = _hint_members(hint)
    word = value.strip().lower()
    for m in members:
        if m is type(None) and word in _NONE_WORDS:
            return None
        if m is bool and word in _BOOL_WORDS:
            return _BOOL_WORDS[word]
        if m in (int, float):
            try:
                return m(value)
            except ValueError:
                pass
        if typing.get_origin(m) is typing.Literal and value in typing.get_args(m):
            return value
    if str in members:
        return value
    raise ValueError(f"config field {name!r}: cannot read {value!r} as {hint}")


def config_fields(cls: type) -> dict[str, Any]:
    """Return the fields of a planner config dataclass with their defaults.

    Parameters
    ----------
    cls : type
        A dataclass whose fields all have defaults.

    Returns
    -------
    dict
        Field name to default value, in declaration order.
    """
    return {f.name: (f.default if f.default is not dataclasses.MISSING else f.default_factory())
            for f in dataclasses.fields(cls)}


def resolve_config(cls: type, overrides: Mapping[str, Any] | Any | None = None):
    """Build a planner config from its defaults and the given overrides.

    Parameters
    ----------
    cls : type
        The planner's config dataclass.  Every field has a default.
    overrides : mapping or cls instance, optional
        Field values to change.  Strings are converted to the field's type
        (``"500"`` to 500, ``"true"`` to True, ``"none"`` to None), so values
        from the command line can be passed as they are.  An instance of
        `cls` is returned unchanged.  None means all defaults.

    Returns
    -------
    cls instance
        The resolved config.

    Raises
    ------
    ValueError
        If a key is not a field of `cls` (the message lists the valid
        fields), or a string cannot be converted to the field's type, or the
        config's own validation rejects a value.

    Examples
    --------
    >>> from planners.rrt_connect import RrtConnectConfig
    >>> resolve_config(RrtConnectConfig, {"max_samples": "100"}).max_samples
    100
    """
    if isinstance(overrides, cls):
        return overrides
    overrides = dict(overrides or {})
    valid = [f.name for f in dataclasses.fields(cls)]
    unknown = sorted(set(overrides) - set(valid))
    if unknown:
        raise ValueError(f"{cls.__name__}: unknown option(s) {unknown}; valid options are {valid}")
    hints = typing.get_type_hints(cls)
    values = {k: _coerce(k, v, hints[k]) for k, v in overrides.items()}
    try:
        return cls(**values)
    except (TypeError, ValueError) as e:
        raise ValueError(f"{cls.__name__}: {e}") from e

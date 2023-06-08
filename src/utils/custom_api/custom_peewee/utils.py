# -*- coding: utf-8 -*-
# @Time: 2023/6/8 17:13
import builtins
import warnings
from collections import Callable
from inspect import isclass
from itertools import zip_longest

from .constants import *


callable_ = lambda c: isinstance(c, Callable)
text_type = str
bytes_type = bytes
buffer_type = memoryview
basestring = str
long = int
multi_types = (list, tuple, frozenset, set, range)
print_t = getattr(builtins, 'print')


def __deprecated__(s):
    warnings.warn(s, DeprecationWarning)


def reraise(tp, value, tb=None):
    if value.__traceback__ is not tb:
        raise value.with_traceback(tb)
    raise value


def with_metaclass(meta, base=object):
    return meta(MODEL_BASE, (base,), {})


def merge_dict(source, overrides):
    merged = source.copy()
    if overrides:
        merged.update(overrides)
    return merged


def quote(path, quote_chars):
    if len(path) == 1:
        return path[0].join(quote_chars)
    return '.'.join([part.join(quote_chars) for part in path])


def is_model(o):
    from .pw import Model
    return isclass(o) and issubclass(o, Model)


def ensure_tuple(value):
    if value is not None:
        return value if isinstance(value, (list, tuple)) else (value,)


def ensure_entity(value):
    from .pw import Node, Entity
    if value is not None:
        return value if isinstance(value, Node) else Entity(value)


def make_snake_case(s):
    first = SNAKE_CASE_STEP1.sub(r'\1_\2', s)
    return SNAKE_CASE_STEP2.sub(r'\1_\2', first).lower()


def chunked(it, n):
    marker = object()
    for group in (list(g) for g in zip_longest(*[iter(it)] * n,
                                               fillvalue=marker)):
        if group[-1] is marker:
            del group[group.index(marker):]
        yield group


def safe_python_value(conv_func):
    def validate(value):
        try:
            return conv_func(value)
        except (TypeError, ValueError):
            return value

    return validate


__all__ = [
    'callable_',
    'text_type',
    'bytes_type',
    'buffer_type',
    'basestring',
    'long',
    'multi_types',
    'print_t',
    '__deprecated__',
    'reraise',
    'ensure_entity',
    'ensure_tuple',
    'make_snake_case',
    'chunked',
    'with_metaclass',
    'is_model',
    'merge_dict',
    'quote',
    'safe_python_value',
]

# -*- coding: utf-8 -*-
# @Time: 2023/6/8 17:10
import calendar
import collections
import datetime
import decimal
import hashlib
import itertools
import operator
import re
import socket
import struct
import time
import uuid
from bisect import bisect_left
from bisect import bisect_right
from collections import Mapping
from contextlib import contextmanager
from copy import deepcopy
from functools import reduce
from functools import wraps

from .constants import *
from .exc import *
from .utils import *


class _callable_context_manager(object):
    def __call__(self, fn):
        @wraps(fn)
        def inner(*args, **kwargs):
            with self:
                return fn(*args, **kwargs)

        return inner


class ModelDescriptor:
    pass


# SQL Generation.


class AliasManager(object):
    __slots__ = ('_counter', '_current_index', '_mapping')

    def __init__(self):
        # A list of dictionaries containing mappings at various depths.
        self._counter = 0
        self._current_index = 0
        self._mapping = []
        self.push()

    @property
    def mapping(self):
        return self._mapping[self._current_index - 1]

    def add(self, source):
        if source not in self.mapping:
            self._counter += 1
            self[source] = 't%d' % self._counter
        return self.mapping[source]

    def get(self, source, any_depth=False):
        if any_depth:
            for idx in reversed(range(self._current_index)):
                if source in self._mapping[idx]:
                    return self._mapping[idx][source]
        return self.add(source)

    def __getitem__(self, source):
        return self.get(source)

    def __setitem__(self, source, alias):
        self.mapping[source] = alias

    def push(self):
        self._current_index += 1
        if self._current_index > len(self._mapping):
            self._mapping.append({})

    def pop(self):
        if self._current_index == 1:
            raise ValueError('Cannot pop() from empty alias manager.')
        self._current_index -= 1


class State(collections.namedtuple('_State', ('scope', 'parentheses',
                                              'settings'))):
    def __new__(cls, scope=SCOPE_NORMAL, parentheses=False, **kwargs):
        return super(State, cls).__new__(cls, scope, parentheses, kwargs)

    def __call__(self, scope=None, parentheses=None, **kwargs):
        # Scope and settings are "inherited" (parentheses is not, however).
        scope = self.scope if scope is None else scope

        # Try to avoid unnecessary dict copying.
        if kwargs and self.settings:
            settings = self.settings.copy()  # Copy original settings dict.
            settings.update(kwargs)  # Update copy with overrides.
        elif kwargs:
            settings = kwargs
        else:
            settings = self.settings
        return State(scope, parentheses, **settings)

    def __getattr__(self, attr_name):
        return self.settings.get(attr_name)


def __scope_context__(scope):
    @contextmanager
    def inner(self, **kwargs):
        with self(scope=scope, **kwargs):
            yield self

    return inner


class Context(object):
    __slots__ = ('stack', '_sql', '_values', 'alias_manager', 'state')
    returning_clause = False

    def __init__(self, **settings):
        self.stack = []
        self._sql = []
        self._values = []
        self.alias_manager = AliasManager()
        self.state = State(**settings)

    def as_new(self):
        return Context(**self.state.settings)

    def column_sort_key(self, item):
        return item[0].get_sort_key(self)

    @property
    def scope(self):
        return self.state.scope

    @property
    def parentheses(self):
        return self.state.parentheses

    @property
    def subquery(self):
        return self.state.subquery

    def __call__(self, **overrides):
        if overrides and overrides.get('scope') == self.scope:
            del overrides['scope']

        self.stack.append(self.state)
        self.state = self.state(**overrides)
        return self

    scope_normal = __scope_context__(SCOPE_NORMAL)
    scope_source = __scope_context__(SCOPE_SOURCE)
    scope_values = __scope_context__(SCOPE_VALUES)
    scope_cte = __scope_context__(SCOPE_CTE)
    scope_column = __scope_context__(SCOPE_COLUMN)

    def __enter__(self):
        if self.parentheses:
            self.literal('(')
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.parentheses:
            self.literal(')')
        self.state = self.stack.pop()

    @contextmanager
    def push_alias(self):
        self.alias_manager.push()
        yield
        self.alias_manager.pop()

    def sql(self, obj):
        if isinstance(obj, (Node, Context)):
            return obj.__sql__(self)
        elif is_model(obj):
            return obj._meta.table.__sql__(self)
        else:
            return self.sql(Value(obj))

    def literal(self, keyword):
        self._sql.append(keyword)
        return self

    def value(self, value, converter=None, add_param=True):
        if converter:
            value = converter(value)
        elif converter is None and self.state.converter:
            # Explicitly check for None so that "False" can be used to signify
            # that no conversion should be applied.
            value = self.state.converter(value)

        if isinstance(value, Node):
            with self(converter=None):
                return self.sql(value)
        elif is_model(value):
            # Under certain circumstances, we could end-up treating a model-
            # class itself as a value. This check ensures that we drop the
            # table alias into the query instead of trying to parameterize a
            # model (for instance, passing a model as a function argument).
            with self.scope_column():
                return self.sql(value)

        self._values.append(value)
        return self.literal(self.state.param or '?') if add_param else self

    def __sql__(self, ctx):
        ctx._sql.extend(self._sql)
        ctx._values.extend(self._values)
        return ctx

    def parse(self, node):
        return self.sql(node).query()

    def query(self):
        return ''.join(self._sql), self._values

    def extract_date(self, date_part, date_field):
        return fn.date_part(date_part, date_field, python_value=int)

    def truncate_date(self, date_part, date_field):
        return fn.date_trunc(date_part, date_field,
                             python_value=simple_date_time)

    def to_timestamp(self, date_field):
        return fn.strftime('%s', date_field).cast('integer')

    def from_timestamp(self, date_field):
        return fn.datetime(date_field, 'unixepoch')

    def get_binary_type(self):
        try:
            return bytes
        except ImportError:
            raise RuntimeError("")


def query_to_string(query):
    # NOTE: this function is not exported by default as it might be misused --
    # and this misuse could lead to sql injection vulnerabilities. This
    # function is intended for debugging or logging purposes ONLY.
    context = getattr(query, '_context', None)
    if context is not None:
        ctx = context
    else:
        ctx = Context()

    sql, params = ctx.sql(query).query()
    if not params:
        return sql

    param = ctx.state.param or '?'
    if param == '?':
        sql = sql.replace('?', '%s')

    return sql % tuple(map(_query_val_transform, params))


def _query_val_transform(v):
    # Interpolate parameters.
    if isinstance(v, (text_type, datetime.datetime, datetime.date,
                      datetime.time)):
        v = "'%s'" % v
    elif isinstance(v, bytes_type):
        try:
            v = v.decode('utf8')
        except UnicodeDecodeError:
            v = v.decode('raw_unicode_escape')
        v = "'%s'" % v
    elif isinstance(v, int):
        v = '%s' % int(v)  # Also handles booleans -> 1 or 0.
    elif v is None:
        v = 'NULL'
    else:
        v = str(v)
    return v


# AST.


class Node(object):
    _coerce = True

    def clone(self):
        obj = self.__class__.__new__(self.__class__)
        obj.__dict__ = self.__dict__.copy()
        return obj

    def __sql__(self, ctx):
        raise NotImplementedError

    @staticmethod
    def copy(method):
        def inner(self, *args, **kwargs):
            clone = self.clone()
            method(clone, *args, **kwargs)
            return clone

        return inner

    def coerce(self, _coerce=True):
        if _coerce != self._coerce:
            clone = self.clone()
            clone._coerce = _coerce
            return clone
        return self

    def is_alias(self):
        return False

    def unwrap(self):
        return self


class ColumnFactory(object):
    __slots__ = ('node',)

    def __init__(self, node):
        self.node = node

    def __getattr__(self, attr):
        return Column(self.node, attr)


class _DynamicColumn(object):
    __slots__ = ()

    def __get__(self, instance, instance_type=None):
        if instance is not None:
            return ColumnFactory(instance)  # Implements __getattr__().
        return self


class _ExplicitColumn(object):
    __slots__ = ()

    def __get__(self, instance, instance_type=None):
        if instance is not None:
            raise AttributeError(
                '%s specifies columns explicitly, and does not support '
                'dynamic column lookups.' % instance)
        return self


class Source(Node):
    c = _DynamicColumn()

    def __init__(self, alias=None):
        super(Source, self).__init__()
        self._alias = alias

    @Node.copy
    def alias(self, name):
        self._alias = name

    def select(self, *columns):
        if not columns:
            columns = (SQL('*'),)
        return Select((self,), columns)

    def join(self, dest, join_type=JOIN.INNER, on=None):
        return Join(self, dest, join_type, on)

    def left_outer_join(self, dest, on=None):
        return Join(self, dest, JOIN.LEFT_OUTER, on)

    def cte(self, name, recursive=False, columns=None, materialized=None):
        return CTE(name, self, recursive=recursive, columns=columns,
                   materialized=materialized)

    def get_sort_key(self, ctx):
        if self._alias:
            return (self._alias,)
        return (ctx.alias_manager[self],)

    def apply_alias(self, ctx):
        # If we are defining the source, include the "AS alias" declaration. An
        # alias is created for the source if one is not already defined.
        if ctx.scope == SCOPE_SOURCE:
            if self._alias:
                ctx.alias_manager[self] = self._alias
            ctx.literal(' AS ').sql(Entity(ctx.alias_manager[self]))
        return ctx

    def apply_column(self, ctx):
        if self._alias:
            ctx.alias_manager[self] = self._alias
        return ctx.sql(Entity(ctx.alias_manager[self]))


class _HashableSource(object):
    def __init__(self, *args, **kwargs):
        super(_HashableSource, self).__init__(*args, **kwargs)
        self._update_hash()

    @Node.copy
    def alias(self, name):
        self._alias = name
        self._update_hash()

    def _update_hash(self):
        self._hash = self._get_hash()

    def _get_hash(self):
        return hash((self.__class__, self._path, self._alias))

    def __hash__(self):
        return self._hash

    def __eq__(self, other):
        if isinstance(other, _HashableSource):
            return self._hash == other._hash
        return Expression(self, OP.EQ, other)

    def __ne__(self, other):
        if isinstance(other, _HashableSource):
            return self._hash != other._hash
        return Expression(self, OP.NE, other)

    def _e(op):
        def inner(self, rhs):
            return Expression(self, op, rhs)

        return inner

    __lt__ = _e(OP.LT)
    __le__ = _e(OP.LTE)
    __gt__ = _e(OP.GT)
    __ge__ = _e(OP.GTE)


def __bind_context__(meth):
    @wraps(meth)
    def inner(self, *args, **kwargs):
        result = meth(self, *args, **kwargs)
        if self._context:
            return result.bind(self._context)
        return result

    return inner


def __join__(join_type=JOIN.INNER, inverted=False):
    def method(self, other):
        if inverted:
            self, other = other, self
        return Join(self, other, join_type=join_type)

    return method


class BaseTable(Source):
    __and__ = __join__(JOIN.INNER)
    __add__ = __join__(JOIN.LEFT_OUTER)
    __sub__ = __join__(JOIN.RIGHT_OUTER)
    __or__ = __join__(JOIN.FULL_OUTER)
    __mul__ = __join__(JOIN.CROSS)
    __rand__ = __join__(JOIN.INNER, inverted=True)
    __radd__ = __join__(JOIN.LEFT_OUTER, inverted=True)
    __rsub__ = __join__(JOIN.RIGHT_OUTER, inverted=True)
    __ror__ = __join__(JOIN.FULL_OUTER, inverted=True)
    __rmul__ = __join__(JOIN.CROSS, inverted=True)


class _BoundTableContext(_callable_context_manager):
    def __init__(self, table, context):
        self.table = table
        self.context = context

    def __enter__(self):
        self._orig_context = self.table._context
        self.table.bind(self.context)
        if self.table._model is not None:
            self.table._model.bind(self.context)
        return self.table

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.table.bind(self._orig_context)
        if self.table._model is not None:
            self.table._model.bind(self._orig_context)


class Table(_HashableSource, BaseTable):
    def __init__(self, name, columns=None, primary_key=None, schema=None,
                 alias=None, _model=None, _context=None):
        self.__name__ = name
        self._columns = columns
        self._primary_key = primary_key
        self._schema = schema
        self._path = (schema, name) if schema else (name,)
        self._model = _model
        self._context = _context
        super(Table, self).__init__(alias=alias)

        # Allow tables to restrict what columns are available.
        if columns is not None:
            self.c = _ExplicitColumn()
            for column in columns:
                setattr(self, column, Column(self, column))

        if primary_key:
            col_src = self if self._columns else self.c
            self.primary_key = getattr(col_src, primary_key)
        else:
            self.primary_key = None

    def clone(self):
        # Ensure a deep copy of the column instances.
        return Table(
            self.__name__,
            columns=self._columns,
            primary_key=self._primary_key,
            schema=self._schema,
            alias=self._alias,
            _model=self._model,
            _context=self._context)

    def bind(self, context=None):
        self._context = context
        return self

    def bind_ctx(self, context=None):
        return _BoundTableContext(self, context)

    def _get_hash(self):
        return hash((self.__class__, self._path, self._alias, self._model))

    @__bind_context__
    def select(self, *columns):
        if not columns and self._columns:
            columns = [Column(self, column) for column in self._columns]
        return Select((self,), columns)

    @__bind_context__
    def insert(self, insert=None, columns=None, **kwargs):
        if kwargs:
            insert = {} if insert is None else insert
            src = self if self._columns else self.c
            for key, value in kwargs.items():
                insert[getattr(src, key)] = value
        return Insert(self, insert=insert, columns=columns)

    @__bind_context__
    def replace(self, insert=None, columns=None, **kwargs):
        return (self
                .insert(insert=insert, columns=columns)
                .on_conflict('REPLACE'))

    @__bind_context__
    def update(self, update=None, **kwargs):
        if kwargs:
            update = {} if update is None else update
            for key, value in kwargs.items():
                src = self if self._columns else self.c
                update[getattr(src, key)] = value
        return Update(self, update=update)

    @__bind_context__
    def delete(self):
        return Delete(self)

    def __sql__(self, ctx):
        if ctx.scope == SCOPE_VALUES:
            # Return the quoted table name.
            return ctx.sql(Entity(*self._path))

        if self._alias:
            ctx.alias_manager[self] = self._alias

        if ctx.scope == SCOPE_SOURCE:
            # Define the table and its alias.
            return self.apply_alias(ctx.sql(Entity(*self._path)))
        else:
            # Refer to the table using the alias.
            return self.apply_column(ctx)


class Join(BaseTable):
    def __init__(self, lhs, rhs, join_type=JOIN.INNER, on=None, alias=None):
        super(Join, self).__init__(alias=alias)
        self.lhs = lhs
        self.rhs = rhs
        self.join_type = join_type
        self._on = on

    def on(self, predicate):
        self._on = predicate
        return self

    def __sql__(self, ctx):
        (ctx
         .sql(self.lhs)
         .literal(' %s ' % self.join_type)
         .sql(self.rhs))
        if self._on is not None:
            ctx.literal(' ON ').sql(self._on)
        return ctx


class ValuesList(_HashableSource, BaseTable):
    def __init__(self, values, columns=None, alias=None):
        self._values = values
        self._columns = columns
        super(ValuesList, self).__init__(alias=alias)

    def _get_hash(self):
        return hash((self.__class__, id(self._values), self._alias))

    @Node.copy
    def columns(self, *names):
        self._columns = names

    def __sql__(self, ctx):
        if self._alias:
            ctx.alias_manager[self] = self._alias

        if ctx.scope == SCOPE_SOURCE or ctx.scope == SCOPE_NORMAL:
            with ctx(parentheses=not ctx.parentheses):
                ctx = (ctx
                    .literal('VALUES ')
                    .sql(CommaNodeList([
                    EnclosedNodeList(row) for row in self._values])))

            if ctx.scope == SCOPE_SOURCE:
                ctx.literal(' AS ').sql(Entity(ctx.alias_manager[self]))
                if self._columns:
                    entities = [Entity(c) for c in self._columns]
                    ctx.sql(EnclosedNodeList(entities))
        else:
            ctx.sql(Entity(ctx.alias_manager[self]))

        return ctx


class CTE(_HashableSource, Source):
    def __init__(self, name, query, recursive=False, columns=None,
                 materialized=None):
        self._alias = name
        self._query = query
        self._recursive = recursive
        self._materialized = materialized
        if columns is not None:
            columns = [Entity(c) if isinstance(c, basestring) else c
                       for c in columns]
        self._columns = columns
        query._cte_list = ()
        super(CTE, self).__init__(alias=name)

    def select_from(self, *columns):
        if not columns:
            raise ValueError('select_from() must specify one or more columns '
                             'from the CTE to select.')

        query = (Select((self,), columns)
                 .with_cte(self)
                 .bind(self._query._context))
        try:
            query = query.objects(self._query.model)
        except AttributeError:
            pass
        return query

    def _get_hash(self):
        return hash((self.__class__, self._alias, id(self._query)))

    def union_all(self, rhs):
        clone = self._query.clone()
        return CTE(self._alias, clone + rhs, self._recursive, self._columns)

    __add__ = union_all

    def union(self, rhs):
        clone = self._query.clone()
        return CTE(self._alias, clone | rhs, self._recursive, self._columns)

    __or__ = union

    def __sql__(self, ctx):
        if ctx.scope != SCOPE_CTE:
            return ctx.sql(Entity(self._alias))

        with ctx.push_alias():
            ctx.alias_manager[self] = self._alias
            ctx.sql(Entity(self._alias))

            if self._columns:
                ctx.literal(' ').sql(EnclosedNodeList(self._columns))
            ctx.literal(' AS ')

            if self._materialized:
                ctx.literal('MATERIALIZED ')
            elif self._materialized is False:
                ctx.literal('NOT MATERIALIZED ')

            with ctx.scope_normal(parentheses=True):
                ctx.sql(self._query)
        return ctx


class ColumnBase(Node):
    _converter = None

    @Node.copy
    def converter(self, converter=None):
        self._converter = converter

    def alias(self, alias):
        if alias:
            return Alias(self, alias)
        return self

    def unalias(self):
        return self

    def cast(self, as_type):
        return Cast(self, as_type)

    def asc(self, collation=None, nulls=None):
        return Asc(self, collation=collation, nulls=nulls)

    __pos__ = asc

    def desc(self, collation=None, nulls=None):
        return Desc(self, collation=collation, nulls=nulls)

    __neg__ = desc

    def __invert__(self):
        return Negated(self)

    def _e(op, inv=False):
        """
        Lightweight factory which returns a method that builds an Expression
        consisting of the left-hand and right-hand operands, using `op`.
        """

        def inner(self, rhs):
            if inv:
                return Expression(rhs, op, self)
            return Expression(self, op, rhs)

        return inner

    __and__ = _e(OP.AND)
    __or__ = _e(OP.OR)

    __add__ = _e(OP.ADD)
    __sub__ = _e(OP.SUB)
    __mul__ = _e(OP.MUL)
    __div__ = __truediv__ = _e(OP.DIV)
    __xor__ = _e(OP.XOR)
    __radd__ = _e(OP.ADD, inv=True)
    __rsub__ = _e(OP.SUB, inv=True)
    __rmul__ = _e(OP.MUL, inv=True)
    __rdiv__ = __rtruediv__ = _e(OP.DIV, inv=True)
    __rand__ = _e(OP.AND, inv=True)
    __ror__ = _e(OP.OR, inv=True)
    __rxor__ = _e(OP.XOR, inv=True)

    def __eq__(self, rhs):
        op = OP.IS if rhs is None else OP.EQ
        return Expression(self, op, rhs)

    def __ne__(self, rhs):
        op = OP.IS_NOT if rhs is None else OP.NE
        return Expression(self, op, rhs)

    __lt__ = _e(OP.LT)
    __le__ = _e(OP.LTE)
    __gt__ = _e(OP.GT)
    __ge__ = _e(OP.GTE)
    __lshift__ = _e(OP.IN)
    __rshift__ = _e(OP.IS)
    __mod__ = _e(OP.LIKE)
    __pow__ = _e(OP.ILIKE)

    bin_and = _e(OP.BIN_AND)
    bin_or = _e(OP.BIN_OR)
    in_ = _e(OP.IN)
    not_in = _e(OP.NOT_IN)
    regexp = _e(OP.REGEXP)

    # Special expressions.
    def is_null(self, is_null=True):
        op = OP.IS if is_null else OP.IS_NOT
        return Expression(self, op, None)

    def _escape_like_expr(self, s, template):
        if s.find('_') >= 0 or s.find('%') >= 0 or s.find('\\') >= 0:
            s = s.replace('\\', '\\\\').replace('_', '\\_').replace('%', '\\%')
            return NodeList((template % s, SQL('ESCAPE'), '\\'))
        return template % s

    def contains(self, rhs):
        if isinstance(rhs, Node):
            rhs = Expression('%', OP.CONCAT,
                             Expression(rhs, OP.CONCAT, '%'))
        else:
            rhs = self._escape_like_expr(rhs, '%%%s%%')
        return Expression(self, OP.ILIKE, rhs)

    def startswith(self, rhs):
        if isinstance(rhs, Node):
            rhs = Expression(rhs, OP.CONCAT, '%')
        else:
            rhs = self._escape_like_expr(rhs, '%s%%')
        return Expression(self, OP.ILIKE, rhs)

    def endswith(self, rhs):
        if isinstance(rhs, Node):
            rhs = Expression('%', OP.CONCAT, rhs)
        else:
            rhs = self._escape_like_expr(rhs, '%%%s')
        return Expression(self, OP.ILIKE, rhs)

    def between(self, lo, hi):
        return Expression(self, OP.BETWEEN, NodeList((lo, SQL('AND'), hi)))

    def concat(self, rhs):
        return StringExpression(self, OP.CONCAT, rhs)

    def regexp(self, rhs):
        return Expression(self, OP.REGEXP, rhs)

    def iregexp(self, rhs):
        return Expression(self, OP.IREGEXP, rhs)

    def __getitem__(self, item):
        if isinstance(item, slice):
            if item.start is None or item.stop is None:
                raise ValueError('BETWEEN range must have both a start- and '
                                 'end-point.')
            return self.between(item.start, item.stop)
        return self == item

    def distinct(self):
        return NodeList((SQL('DISTINCT'), self))

    def collate(self, collation):
        return NodeList((self, SQL('COLLATE %s' % collation)))

    def get_sort_key(self, ctx):
        return ()


class Column(ColumnBase):
    def __init__(self, source, name):
        self.source = source
        self.name = name

    def get_sort_key(self, ctx):
        if ctx.scope == SCOPE_VALUES:
            return (self.name,)
        else:
            return self.source.get_sort_key(ctx) + (self.name,)

    def __hash__(self):
        return hash((self.source, self.name))

    def __sql__(self, ctx):
        if ctx.scope == SCOPE_VALUES:
            return ctx.sql(Entity(self.name))
        else:
            with ctx.scope_column():
                return ctx.sql(self.source).literal('.').sql(Entity(self.name))


class WrappedNode(ColumnBase):
    def __init__(self, node):
        self.node = node
        self._coerce = getattr(node, '_coerce', True)
        self._converter = getattr(node, '_converter', None)

    def is_alias(self):
        return self.node.is_alias()

    def unwrap(self):
        return self.node.unwrap()


class EntityFactory(object):
    __slots__ = ('node',)

    def __init__(self, node):
        self.node = node

    def __getattr__(self, attr):
        return Entity(self.node, attr)


class _DynamicEntity(object):
    __slots__ = ()

    def __get__(self, instance, instance_type=None):
        if instance is not None:
            return EntityFactory(instance._alias)  # Implements __getattr__().
        return self


class Alias(WrappedNode):
    c = _DynamicEntity()

    def __init__(self, node, alias):
        super(Alias, self).__init__(node)
        self._alias = alias

    def __hash__(self):
        return hash(self._alias)

    def alias(self, alias=None):
        if alias is None:
            return self.node
        else:
            return Alias(self.node, alias)

    def unalias(self):
        return self.node

    def is_alias(self):
        return True

    def __sql__(self, ctx):
        if ctx.scope == SCOPE_SOURCE:
            return (ctx
                    .sql(self.node)
                    .literal(' AS ')
                    .sql(Entity(self._alias)))
        else:
            return ctx.sql(Entity(self._alias))


class Negated(WrappedNode):
    def __invert__(self):
        return self.node

    def __sql__(self, ctx):
        return ctx.literal('NOT ').sql(self.node)


class BitwiseMixin(object):
    def __and__(self, other):
        return self.bin_and(other)

    def __or__(self, other):
        return self.bin_or(other)

    def __sub__(self, other):
        return self.bin_and(other.bin_negated())

    def __invert__(self):
        return BitwiseNegated(self)


class BitwiseNegated(BitwiseMixin, WrappedNode):
    def __invert__(self):
        return self.node

    def __sql__(self, ctx):
        if ctx.state.operations:
            op_sql = ctx.state.operations.get(self.op, self.op)
        else:
            op_sql = self.op
        return ctx.literal(op_sql).sql(self.node)


class Value(ColumnBase):
    def __init__(self, value, converter=None, unpack=True):
        self.value = value
        self.converter = converter
        self.multi = unpack and isinstance(self.value, multi_types)
        if self.multi:
            self.values = []
            for item in self.value:
                if isinstance(item, Node):
                    self.values.append(item)
                else:
                    self.values.append(Value(item, self.converter))

    def __sql__(self, ctx):
        if self.multi:
            # For multi-part values (e.g. lists of IDs).
            return ctx.sql(EnclosedNodeList(self.values))

        return ctx.value(self.value, self.converter)


def AsIs(value):
    return Value(value, unpack=False)


class Cast(WrappedNode):
    def __init__(self, node, cast):
        super(Cast, self).__init__(node)
        self._cast = cast
        self._coerce = False

    def __sql__(self, ctx):
        return (ctx
                .literal('CAST(')
                .sql(self.node)
                .literal(' AS %s)' % self._cast))


class Ordering(WrappedNode):
    def __init__(self, node, direction, collation=None, nulls=None):
        super(Ordering, self).__init__(node)
        self.direction = direction
        self.collation = collation
        self.nulls = nulls
        if nulls and nulls.lower() not in ('first', 'last'):
            raise ValueError('Ordering nulls= parameter must be "first" or '
                             '"last", got: %s' % nulls)

    def collate(self, collation=None):
        return Ordering(self.node, self.direction, collation)

    def _null_ordering_case(self, nulls):
        if nulls.lower() == 'last':
            ifnull, notnull = 1, 0
        elif nulls.lower() == 'first':
            ifnull, notnull = 0, 1
        else:
            raise ValueError('unsupported value for nulls= ordering.')
        return Case(None, ((self.node.is_null(), ifnull),), notnull)

    def __sql__(self, ctx):
        if self.nulls and not ctx.state.nulls_ordering:
            ctx.sql(self._null_ordering_case(self.nulls)).literal(', ')

        ctx.sql(self.node).literal(' %s' % self.direction)
        if self.collation:
            ctx.literal(' COLLATE %s' % self.collation)
        if self.nulls and ctx.state.nulls_ordering:
            ctx.literal(' NULLS %s' % self.nulls)
        return ctx


def Asc(node, collation=None, nulls=None):
    return Ordering(node, 'ASC', collation, nulls)


def Desc(node, collation=None, nulls=None):
    return Ordering(node, 'DESC', collation, nulls)


class Expression(ColumnBase):
    def __init__(self, lhs, op, rhs, flat=False):
        self.lhs = lhs
        self.op = op
        self.rhs = rhs
        self.flat = flat

    def __sql__(self, ctx):
        overrides = {'parentheses': not self.flat, 'in_expr': True}

        # First attempt to unwrap the node on the left-hand-side, so that we
        # can get at the underlying Field if one is present.
        node = raw_node = self.lhs
        if isinstance(raw_node, WrappedNode):
            node = raw_node.unwrap()

        # Set up the appropriate converter if we have a field on the left side.
        if isinstance(node, Field) and raw_node._coerce:
            overrides['converter'] = node.db_value
        else:
            overrides['converter'] = None

        if ctx.state.operations:
            op_sql = ctx.state.operations.get(self.op, self.op)
        else:
            op_sql = self.op

        with ctx(**overrides):
            # Postgresql reports an error for IN/NOT IN (), so convert to
            # the equivalent boolean expression.
            op_in = self.op == OP.IN or self.op == OP.NOT_IN
            if op_in and ctx.as_new().parse(self.rhs)[0] == '()':
                return ctx.literal('0 = 1' if self.op == OP.IN else '1 = 1')

            return (ctx
                    .sql(self.lhs)
                    .literal(' %s ' % op_sql)
                    .sql(self.rhs))


class StringExpression(Expression):
    def __add__(self, rhs):
        return self.concat(rhs)

    def __radd__(self, lhs):
        return StringExpression(lhs, OP.CONCAT, self)


class Entity(ColumnBase):
    def __init__(self, *path):
        self._path = [part.replace('"', '""') for part in path if part]

    def __getattr__(self, attr):
        return Entity(*self._path + [attr])

    def get_sort_key(self, ctx):
        return tuple(self._path)

    def __hash__(self):
        return hash((self.__class__.__name__, tuple(self._path)))

    def __sql__(self, ctx):
        return ctx.literal(quote(self._path, ctx.state.quote or '""'))


class SQL(ColumnBase):
    def __init__(self, sql, params=None):
        self.sql = sql
        self.params = params

    def __sql__(self, ctx):
        ctx.literal(self.sql)
        if self.params:
            for param in self.params:
                ctx.value(param, False, add_param=False)
        return ctx


def Check(constraint):
    return SQL('CHECK (%s)' % constraint)


class Function(ColumnBase):
    def __init__(self, name, arguments, coerce=True, python_value=None):
        self.name = name
        self.arguments = arguments
        self._filter = None
        self._order_by = None
        self._python_value = python_value
        if name and name.lower() in ('sum', 'count', 'cast'):
            self._coerce = False
        else:
            self._coerce = coerce

    def __getattr__(self, attr):
        def decorator(*args, **kwargs):
            return Function(attr, args, **kwargs)

        return decorator

    @Node.copy
    def filter(self, where=None):
        self._filter = where

    @Node.copy
    def order_by(self, *ordering):
        self._order_by = ordering

    @Node.copy
    def python_value(self, func=None):
        self._python_value = func

    def over(self, partition_by=None, order_by=None, start=None, end=None,
             frame_type=None, window=None, exclude=None):
        if isinstance(partition_by, Window) and window is None:
            window = partition_by

        if window is not None:
            node = WindowAlias(window)
        else:
            node = Window(partition_by=partition_by, order_by=order_by,
                          start=start, end=end, frame_type=frame_type,
                          exclude=exclude, _inline=True)
        return NodeList((self, SQL('OVER'), node))

    def __sql__(self, ctx):
        ctx.literal(self.name)
        if not len(self.arguments):
            ctx.literal('()')
        else:
            args = self.arguments

            # If this is an ordered aggregate, then we will modify the last
            # argument to append the ORDER BY ... clause. We do this to avoid
            # double-wrapping any expression args in parentheses, as NodeList
            # has a special check (hack) in place to work around this.
            if self._order_by:
                args = list(args)
                args[-1] = NodeList((args[-1], SQL('ORDER BY'),
                                     CommaNodeList(self._order_by)))

            with ctx(in_function=True, function_arg_count=len(self.arguments)):
                ctx.sql(EnclosedNodeList([
                    (arg if isinstance(arg, Node) else Value(arg, False))
                    for arg in args]))

        if self._filter:
            ctx.literal(' FILTER (WHERE ').sql(self._filter).literal(')')
        return ctx


fn = Function(None, None)


class Window(Node):
    # Frame start/end and frame exclusion.
    CURRENT_ROW = SQL('CURRENT ROW')
    GROUP = SQL('GROUP')
    TIES = SQL('TIES')
    NO_OTHERS = SQL('NO OTHERS')

    # Frame types.
    GROUPS = 'GROUPS'
    RANGE = 'RANGE'
    ROWS = 'ROWS'

    def __init__(self, partition_by=None, order_by=None, start=None, end=None,
                 frame_type=None, extends=None, exclude=None, alias=None,
                 _inline=False):
        super(Window, self).__init__()
        if start is not None and not isinstance(start, SQL):
            start = SQL(start)
        if end is not None and not isinstance(end, SQL):
            end = SQL(end)

        self.partition_by = ensure_tuple(partition_by)
        self.order_by = ensure_tuple(order_by)
        self.start = start
        self.end = end
        if self.start is None and self.end is not None:
            raise ValueError('Cannot specify WINDOW end without start.')
        self._alias = alias or 'w'
        self._inline = _inline
        self.frame_type = frame_type
        self._extends = extends
        self._exclude = exclude

    def alias(self, alias=None):
        self._alias = alias or 'w'
        return self

    @Node.copy
    def as_range(self):
        self.frame_type = Window.RANGE

    @Node.copy
    def as_rows(self):
        self.frame_type = Window.ROWS

    @Node.copy
    def as_groups(self):
        self.frame_type = Window.GROUPS

    @Node.copy
    def extends(self, window=None):
        self._extends = window

    @Node.copy
    def exclude(self, frame_exclusion=None):
        if isinstance(frame_exclusion, basestring):
            frame_exclusion = SQL(frame_exclusion)
        self._exclude = frame_exclusion

    @staticmethod
    def following(value=None):
        if value is None:
            return SQL('UNBOUNDED FOLLOWING')
        return SQL('%d FOLLOWING' % value)

    @staticmethod
    def preceding(value=None):
        if value is None:
            return SQL('UNBOUNDED PRECEDING')
        return SQL('%d PRECEDING' % value)

    def __sql__(self, ctx):
        if ctx.scope != SCOPE_SOURCE and not self._inline:
            ctx.literal(self._alias)
            ctx.literal(' AS ')

        with ctx(parentheses=True):
            parts = []
            if self._extends is not None:
                ext = self._extends
                if isinstance(ext, Window):
                    ext = SQL(ext._alias)
                elif isinstance(ext, basestring):
                    ext = SQL(ext)
                parts.append(ext)
            if self.partition_by:
                parts.extend((
                    SQL('PARTITION BY'),
                    CommaNodeList(self.partition_by)))
            if self.order_by:
                parts.extend((
                    SQL('ORDER BY'),
                    CommaNodeList(self.order_by)))
            if self.start is not None and self.end is not None:
                frame = self.frame_type or 'ROWS'
                parts.extend((
                    SQL('%s BETWEEN' % frame),
                    self.start,
                    SQL('AND'),
                    self.end))
            elif self.start is not None:
                parts.extend((SQL(self.frame_type or 'ROWS'), self.start))
            elif self.frame_type is not None:
                parts.append(SQL('%s UNBOUNDED PRECEDING' % self.frame_type))
            if self._exclude is not None:
                parts.extend((SQL('EXCLUDE'), self._exclude))
            ctx.sql(NodeList(parts))
        return ctx


class WindowAlias(Node):
    def __init__(self, window):
        self.window = window

    def alias(self, window_alias):
        self.window._alias = window_alias
        return self

    def __sql__(self, ctx):
        return ctx.literal(self.window._alias or 'w')


class ForUpdate(Node):
    def __init__(self, expr, of=None, nowait=None):
        expr = 'FOR UPDATE' if expr is True else expr
        if expr.lower().endswith('nowait'):
            expr = expr[:-7]  # Strip off the "nowait" bit.
            nowait = True

        self._expr = expr
        if of is not None and not isinstance(of, (list, set, tuple)):
            of = (of,)
        self._of = of
        self._nowait = nowait

    def __sql__(self, ctx):
        ctx.literal(self._expr)
        if self._of is not None:
            ctx.literal(' OF ').sql(CommaNodeList(self._of))
        if self._nowait:
            ctx.literal(' NOWAIT')
        return ctx


def Case(predicate, expression_tuples, default=None):
    clauses = [SQL('CASE')]
    if predicate is not None:
        clauses.append(predicate)
    for expr, value in expression_tuples:
        clauses.extend((SQL('WHEN'), expr, SQL('THEN'), value))
    if default is not None:
        clauses.extend((SQL('ELSE'), default))
    clauses.append(SQL('END'))
    return NodeList(clauses)


class NodeList(ColumnBase):
    def __init__(self, nodes, glue=' ', parens=False):
        self.nodes = nodes
        self.glue = glue
        self.parens = parens
        if parens and len(self.nodes) == 1 and \
                isinstance(self.nodes[0], Expression) and \
                not self.nodes[0].flat:
            # Hack to avoid double-parentheses.
            self.nodes = (self.nodes[0].clone(),)
            self.nodes[0].flat = True

    def __sql__(self, ctx):
        n_nodes = len(self.nodes)
        if n_nodes == 0:
            return ctx.literal('()') if self.parens else ctx
        with ctx(parentheses=self.parens):
            for i in range(n_nodes - 1):
                ctx.sql(self.nodes[i])
                ctx.literal(self.glue)
            ctx.sql(self.nodes[n_nodes - 1])
        return ctx


def CommaNodeList(nodes):
    return NodeList(nodes, ', ')


def EnclosedNodeList(nodes):
    return NodeList(nodes, ', ', True)


class _Namespace(Node):
    __slots__ = ('_name',)

    def __init__(self, name):
        self._name = name

    def __getattr__(self, attr):
        return NamespaceAttribute(self, attr)

    __getitem__ = __getattr__


class NamespaceAttribute(ColumnBase):
    def __init__(self, namespace, attribute):
        self._namespace = namespace
        self._attribute = attribute

    def __sql__(self, ctx):
        return (ctx
                .literal(self._namespace._name + '.')
                .sql(Entity(self._attribute)))


EXCLUDED = _Namespace('EXCLUDED')


class DQ(ColumnBase):
    def __init__(self, **query):
        super(DQ, self).__init__()
        self.query = query
        self._negated = False

    @Node.copy
    def __invert__(self):
        self._negated = not self._negated

    def clone(self):
        node = DQ(**self.query)
        node._negated = self._negated
        return node


#: Represent a row tuple.
Tuple = lambda *a: EnclosedNodeList(a)


class QualifiedNames(WrappedNode):
    def __sql__(self, ctx):
        with ctx.scope_column():
            return ctx.sql(self.node)


def qualify_names(node):
    # Search a node heirarchy to ensure that any column-like objects are
    # referenced using fully-qualified names.
    if isinstance(node, Expression):
        return node.__class__(qualify_names(node.lhs), node.op,
                              qualify_names(node.rhs), node.flat)
    elif isinstance(node, ColumnBase):
        return QualifiedNames(node)
    return node


class OnConflict(Node):
    def __init__(self, action=None, update=None, preserve=None, where=None,
                 conflict_target=None, conflict_where=None,
                 conflict_constraint=None):
        self._action = action
        self._update = update
        self._preserve = ensure_tuple(preserve)
        self._where = where
        if conflict_target is not None and conflict_constraint is not None:
            raise ValueError('only one of "conflict_target" and '
                             '"conflict_constraint" may be specified.')
        self._conflict_target = ensure_tuple(conflict_target)
        self._conflict_where = conflict_where
        self._conflict_constraint = conflict_constraint

    def get_conflict_statement(self, ctx, query):
        return ctx.state.conflict_statement(self, query)

    def get_conflict_update(self, ctx, query):
        return ctx.state.conflict_update(self, query)

    @Node.copy
    def preserve(self, *columns):
        self._preserve = columns

    @Node.copy
    def update(self, _data=None, **kwargs):
        if _data and kwargs and not isinstance(_data, dict):
            raise ValueError('Cannot mix data with keyword arguments in the '
                             'OnConflict update method.')
        _data = _data or {}
        if kwargs:
            _data.update(kwargs)
        self._update = _data

    @Node.copy
    def where(self, *expressions):
        if self._where is not None:
            expressions = (self._where,) + expressions
        self._where = reduce(operator.and_, expressions)

    @Node.copy
    def conflict_target(self, *constraints):
        self._conflict_constraint = None
        self._conflict_target = constraints

    @Node.copy
    def conflict_where(self, *expressions):
        if self._conflict_where is not None:
            expressions = (self._conflict_where,) + expressions
        self._conflict_where = reduce(operator.and_, expressions)

    @Node.copy
    def conflict_constraint(self, constraint):
        self._conflict_constraint = constraint
        self._conflict_target = None


# BASE QUERY INTERFACE.

class BaseQuery(Node):
    default_row_type = ROW.DICT

    def __init__(self, _context=None, **kwargs):
        self._context = _context
        self._row_type = None
        self._constructor = None
        super(BaseQuery, self).__init__(**kwargs)

    def bind(self, context=None):
        self._context = context
        return self

    def clone(self):
        query = super(BaseQuery, self).clone()
        return query

    @Node.copy
    def dicts(self, as_dict=True):
        self._row_type = ROW.DICT if as_dict else None
        return self

    @Node.copy
    def tuples(self, as_tuple=True):
        self._row_type = ROW.TUPLE if as_tuple else None
        return self

    @Node.copy
    def namedtuples(self, as_namedtuple=True):
        self._row_type = ROW.NAMED_TUPLE if as_namedtuple else None
        return self

    @Node.copy
    def objects(self, constructor=None):
        self._row_type = ROW.CONSTRUCTOR if constructor else None
        self._constructor = constructor
        return self

    def __sql__(self, ctx):
        raise NotImplementedError

    def sql(self):
        if self._context:
            context = self._context.as_new()
        else:
            context = Context()
        return context.parse(self)

    async def execute(self, conn=None, context=None):
        # if conn, means in transaction
        if conn:
            return await self._execute(conn, context)
        # if BaseModelSelect and confing slave_conn, use the slave to select
        if isinstance(self, BaseModelSelect):
            slave_conn = self.model._meta.slave_connection_factory
            if slave_conn:
                async with slave_conn() as conn:
                    return await self._execute(conn, context)
        async with self.model._meta.connection_factory() as conn:
            return await self._execute(conn, context)

    async def _execute(self, conn=None, context=None):
        raise NotImplementedError

    def __str__(self):
        return query_to_string(self)


class RawQuery(BaseQuery):
    def __init__(self, sql=None, params=None, **kwargs):
        super(RawQuery, self).__init__(**kwargs)
        self._sql = sql
        self._params = params

    def __sql__(self, ctx):
        ctx.literal(self._sql)
        if self._params:
            for param in self._params:
                ctx.value(param, add_param=False)
        return ctx


class Query(BaseQuery):
    def __init__(self, where=None, order_by=None, limit=None, offset=None,
                 **kwargs):
        super(Query, self).__init__(**kwargs)
        self._where = where
        self._order_by = order_by
        self._limit = limit
        self._offset = offset

        self._cte_list = None

    @Node.copy
    def with_cte(self, *cte_list):
        self._cte_list = cte_list

    @Node.copy
    def where(self, *expressions):
        if self._where is not None:
            expressions = (self._where,) + expressions
        self._where = reduce(operator.and_, expressions)

    @Node.copy
    def orwhere(self, *expressions):
        if self._where is not None:
            expressions = (self._where,) + expressions
        self._where = reduce(operator.or_, expressions)

    @Node.copy
    def order_by(self, *values):
        self._order_by = values

    @Node.copy
    def order_by_extend(self, *values):
        self._order_by = ((self._order_by or ()) + values) or None

    @Node.copy
    def limit(self, value=None):
        self._limit = value

    @Node.copy
    def offset(self, value=None):
        self._offset = value

    @Node.copy
    def paginate(self, page, paginate_by=20):
        if page > 0:
            page -= 1
        self._limit = paginate_by
        self._offset = page * paginate_by

    def _apply_ordering(self, ctx):
        if self._order_by:
            (ctx
             .literal(' ORDER BY ')
             .sql(CommaNodeList(self._order_by)))
        if self._limit is not None or (self._offset is not None and
                                       ctx.state.limit_max):
            limit = ctx.state.limit_max if self._limit is None else self._limit
            ctx.literal(' LIMIT ').sql(limit)
        if self._offset is not None:
            ctx.literal(' OFFSET ').sql(self._offset)
        return ctx

    def __sql__(self, ctx):
        if self._cte_list:
            # The CTE scope is only used at the very beginning of the query,
            # when we are describing the various CTEs we will be using.
            recursive = any(cte._recursive for cte in self._cte_list)

            # Explicitly disable the "subquery" flag here, so as to avoid
            # unnecessary parentheses around subsequent selects.
            with ctx.scope_cte(subquery=False):
                (ctx
                 .literal('WITH RECURSIVE ' if recursive else 'WITH ')
                 .sql(CommaNodeList(self._cte_list))
                 .literal(' '))
        return ctx


def __compound_select__(operation, inverted=False):
    def method(self, other):
        if inverted:
            self, other = other, self
        return CompoundSelectQuery(self, operation, other)

    return method


class SelectQuery(Query):
    union_all = __add__ = __compound_select__('UNION ALL')
    union = __or__ = __compound_select__('UNION')
    intersect = __and__ = __compound_select__('INTERSECT')
    except_ = __sub__ = __compound_select__('EXCEPT')
    __radd__ = __compound_select__('UNION ALL', inverted=True)
    __ror__ = __compound_select__('UNION', inverted=True)
    __rand__ = __compound_select__('INTERSECT', inverted=True)
    __rsub__ = __compound_select__('EXCEPT', inverted=True)

    def select_from(self, *columns):
        if not columns:
            raise ValueError('select_from() must specify one or more columns.')

        query = (Select((self,), columns)
                 .bind(self._context))
        if getattr(self, 'model', None) is not None:
            # Bind to the sub-select's model type, if defined.
            query = query.objects(self.model)
        return query


class SelectBase(_HashableSource, Source, SelectQuery):
    def _get_hash(self):
        return hash((self.__class__, self._alias or id(self)))


# QUERY IMPLEMENTATIONS.


class CompoundSelectQuery(SelectBase):
    def __init__(self, lhs, op, rhs):
        super(CompoundSelectQuery, self).__init__()
        self.lhs = lhs
        self.op = op
        self.rhs = rhs

    def _get_query_key(self):
        return (self.lhs.get_query_key(), self.rhs.get_query_key())

    def _wrap_parens(self, ctx, subq):
        csq_setting = ctx.state.compound_select_parentheses

        if not csq_setting or csq_setting == CSQ_PARENTHESES_NEVER:
            return False
        elif csq_setting == CSQ_PARENTHESES_ALWAYS:
            return True
        elif csq_setting == CSQ_PARENTHESES_UNNESTED:
            if ctx.state.in_expr or ctx.state.in_function:
                # If this compound select query is being used inside an
                # expression, e.g., an IN or EXISTS().
                return False

            # If the query on the left or right is itself a compound select
            # query, then we do not apply parentheses. However, if it is a
            # regular SELECT query, we will apply parentheses.
            return not isinstance(subq, CompoundSelectQuery)

    def __sql__(self, ctx):
        if ctx.scope == SCOPE_COLUMN:
            return self.apply_column(ctx)

        outer_parens = ctx.subquery or (ctx.scope == SCOPE_SOURCE)
        with ctx(parentheses=outer_parens):
            # Should the left-hand query be wrapped in parentheses?
            lhs_parens = self._wrap_parens(ctx, self.lhs)
            with ctx.scope_normal(parentheses=lhs_parens, subquery=False):
                ctx.sql(self.lhs)
            ctx.literal(' %s ' % self.op)
            with ctx.push_alias():
                # Should the right-hand query be wrapped in parentheses?
                rhs_parens = self._wrap_parens(ctx, self.rhs)
                with ctx.scope_normal(parentheses=rhs_parens, subquery=False):
                    ctx.sql(self.rhs)

            # Apply ORDER BY, LIMIT, OFFSET. We use the "values" scope so that
            # entity names are not fully-qualified. This is a bit of a hack, as
            # we're relying on the logic in Column.__sql__() to not fully
            # qualify column names.
            with ctx.scope_values():
                self._apply_ordering(ctx)

        return self.apply_alias(ctx)


class Select(SelectBase):
    def __init__(self, from_list=None, columns=None, group_by=None,
                 having=None, distinct=None, windows=None, for_update=None,
                 for_update_of=None, nowait=None, lateral=None, **kwargs):
        super(Select, self).__init__(**kwargs)
        self._from_list = (list(from_list) if isinstance(from_list, tuple)
                           else from_list) or []
        self._returning = columns
        self._group_by = group_by
        self._having = having
        self._windows = None
        self._for_update = for_update  # XXX: consider reorganizing.
        self._for_update_of = for_update_of
        self._for_update_nowait = nowait
        self._lateral = lateral

        self._distinct = self._simple_distinct = None
        if distinct:
            if isinstance(distinct, bool):
                self._simple_distinct = distinct
            else:
                self._distinct = distinct

    def clone(self):
        clone = super(Select, self).clone()
        if clone._from_list:
            clone._from_list = list(clone._from_list)
        return clone

    @Node.copy
    def columns(self, *columns, **kwargs):
        self._returning = columns

    select = columns

    @Node.copy
    def select_extend(self, *columns):
        self._returning = tuple(self._returning) + columns

    @Node.copy
    def from_(self, *sources):
        self._from_list = list(sources)

    @Node.copy
    def join(self, dest, join_type=JOIN.INNER, on=None):
        if not self._from_list:
            raise ValueError('No sources to join on.')
        item = self._from_list.pop()
        self._from_list.append(Join(item, dest, join_type, on))

    @Node.copy
    def group_by(self, *columns):
        grouping = []
        for column in columns:
            if isinstance(column, Table):
                if not column._columns:
                    raise ValueError('Cannot pass a table to group_by() that '
                                     'does not have columns explicitly '
                                     'declared.')
                grouping.extend([getattr(column, col_name)
                                 for col_name in column._columns])
            else:
                grouping.append(column)
        self._group_by = grouping

    def group_by_extend(self, *values):
        """@Node.copy used from group_by() call"""
        group_by = tuple(self._group_by or ()) + values
        return self.group_by(*group_by)

    @Node.copy
    def having(self, *expressions):
        if self._having is not None:
            expressions = (self._having,) + expressions
        self._having = reduce(operator.and_, expressions)

    @Node.copy
    def distinct(self, *columns):
        if len(columns) == 1 and (columns[0] is True or columns[0] is False):
            self._simple_distinct = columns[0]
        else:
            self._simple_distinct = False
            self._distinct = columns

    @Node.copy
    def window(self, *windows):
        self._windows = windows if windows else None

    @Node.copy
    def for_update(self, for_update=True, of=None, nowait=None):
        if not for_update and (of is not None or nowait):
            for_update = True
        self._for_update = for_update
        self._for_update_of = of
        self._for_update_nowait = nowait

    @Node.copy
    def lateral(self, lateral=True):
        self._lateral = lateral

    def _get_query_key(self):
        return self._alias

    def __sql_selection__(self, ctx, is_subquery=False):
        return ctx.sql(CommaNodeList(self._returning))

    def __sql__(self, ctx):
        if ctx.scope == SCOPE_COLUMN:
            return self.apply_column(ctx)

        if self._lateral and ctx.scope == SCOPE_SOURCE:
            ctx.literal('LATERAL ')

        is_subquery = ctx.subquery
        state = {
            'converter': None,
            'in_function': False,
            'parentheses': is_subquery or (ctx.scope == SCOPE_SOURCE),
            'subquery': True,
        }
        if ctx.state.in_function and ctx.state.function_arg_count == 1:
            state['parentheses'] = False

        with ctx.scope_normal(**state):
            # Defer calling parent SQL until here. This ensures that any CTEs
            # for this query will be properly nested if this query is a
            # sub-select or is used in an expression. See GH#1809 for example.
            super(Select, self).__sql__(ctx)

            ctx.literal('SELECT ')
            if self._simple_distinct or self._distinct is not None:
                ctx.literal('DISTINCT ')
                if self._distinct:
                    (ctx
                     .literal('ON ')
                     .sql(EnclosedNodeList(self._distinct))
                     .literal(' '))

            with ctx.scope_source():
                ctx = self.__sql_selection__(ctx, is_subquery)

            if self._from_list:
                with ctx.scope_source(parentheses=False):
                    ctx.literal(' FROM ').sql(CommaNodeList(self._from_list))

            if self._where is not None:
                ctx.literal(' WHERE ').sql(self._where)

            if self._group_by:
                ctx.literal(' GROUP BY ').sql(CommaNodeList(self._group_by))

            if self._having is not None:
                ctx.literal(' HAVING ').sql(self._having)

            if self._windows is not None:
                ctx.literal(' WINDOW ')
                ctx.sql(CommaNodeList(self._windows))

            # Apply ORDER BY, LIMIT, OFFSET.
            self._apply_ordering(ctx)

            if self._for_update:
                if not ctx.state.for_update:
                    raise ValueError('FOR UPDATE specified but not supported '
                                     'by database.')
                ctx.literal(' ')
                ctx.sql(ForUpdate(self._for_update, self._for_update_of,
                                  self._for_update_nowait))

        # If the subquery is inside a function -or- we are evaluating a
        # subquery on either side of an expression w/o an explicit alias, do
        # not generate an alias + AS clause.
        if ctx.state.in_function or (ctx.state.in_expr and
                                     self._alias is None):
            return ctx

        return self.apply_alias(ctx)


class _WriteQuery(Query):
    def __init__(self, table, returning=None, **kwargs):
        self.table = table
        self._returning = returning
        super(_WriteQuery, self).__init__(**kwargs)

    @Node.copy
    def returning(self, *returning):
        self._returning = returning

    def apply_returning(self, ctx):
        return ctx

    def _set_table_alias(self, ctx):
        ctx.alias_manager[self.table] = self.table.__name__

    def __sql__(self, ctx):
        super(_WriteQuery, self).__sql__(ctx)
        # We explicitly set the table alias to the table's name, which ensures
        # that if a sub-select references a column on the outer table, we won't
        # assign it a new alias (e.g. t2) but will refer to it as table.column.
        self._set_table_alias(ctx)
        return ctx


class Update(_WriteQuery):
    def __init__(self, table, update=None, **kwargs):
        super(Update, self).__init__(table, **kwargs)
        self._update = update
        self._from = None

    @Node.copy
    def from_(self, *sources):
        self._from = sources

    def __sql__(self, ctx):
        super(Update, self).__sql__(ctx)

        with ctx.scope_values(subquery=True):
            ctx.literal('UPDATE ')

            expressions = []
            for k, v in sorted(self._update.items(), key=ctx.column_sort_key):
                if not isinstance(v, Node):
                    if isinstance(k, Field):
                        v = k.to_value(v)
                    else:
                        v = Value(v, unpack=False)
                if not isinstance(v, Value):
                    v = qualify_names(v)
                expressions.append(NodeList((k, SQL('='), v)))

            (ctx
             .sql(self.table)
             .literal(' SET ')
             .sql(CommaNodeList(expressions)))

            if self._from:
                with ctx.scope_source(parentheses=False):
                    ctx.literal(' FROM ').sql(CommaNodeList(self._from))

            if self._where:
                with ctx.scope_normal():
                    ctx.literal(' WHERE ').sql(self._where)
            self._apply_ordering(ctx)
            return self.apply_returning(ctx)

    async def _execute(self, conn=None, _context=None):
        affected_rows = conn.update(*self.sql())
        if affected_rows > 1:
            conn.rollback()
            raise MultiUpdateNotAllowed()
        return affected_rows


class Insert(_WriteQuery):
    SIMPLE = 0
    QUERY = 1
    MULTI = 2

    class DefaultValuesException(Exception):
        pass

    def __init__(self, table, insert=None, columns=None, on_conflict=None,
                 **kwargs):
        super(Insert, self).__init__(table, **kwargs)
        self._insert = insert
        self._columns = columns
        self._on_conflict = on_conflict
        self._query_type = None

    def where(self, *expressions):
        raise NotImplementedError('INSERT queries cannot have a WHERE clause.')

    @Node.copy
    def on_conflict_ignore(self, ignore=True):
        self._on_conflict = OnConflict('IGNORE') if ignore else None

    @Node.copy
    def on_conflict_replace(self, replace=True):
        self._on_conflict = OnConflict('REPLACE') if replace else None

    @Node.copy
    def on_conflict(self, *args, **kwargs):
        self._on_conflict = (OnConflict(*args, **kwargs) if (args or kwargs)
                             else None)

    def _simple_insert(self, ctx):
        if not self._insert:
            raise self.DefaultValuesException('Error: no data to insert.')
        return self._generate_insert((self._insert,), ctx)

    def get_default_data(self):
        return {}

    def get_default_columns(self):
        if self.table._columns:
            return [getattr(self.table, col) for col in self.table._columns
                    if col != self.table._primary_key]

    def _generate_insert(self, insert, ctx):
        rows_iter = iter(insert)
        columns = self._columns

        # Load and organize column defaults (if provided).
        defaults = self.get_default_data()

        # First figure out what columns are being inserted (if they weren't
        # specified explicitly). Resulting columns are normalized and ordered.
        if not columns:
            try:
                row = next(rows_iter)
            except StopIteration:
                raise self.DefaultValuesException('Error: no rows to insert.')

            if not isinstance(row, Mapping):
                columns = self.get_default_columns()
                if columns is None:
                    raise ValueError('Bulk insert must specify columns.')
            else:
                # Infer column names from the dict of data being inserted.
                accum = []
                for column in row:
                    if isinstance(column, basestring):
                        column = getattr(self.table, column)
                    accum.append(column)

                # Add any columns present in the default data that are not
                # accounted for by the dictionary of row data.
                column_set = set(accum)
                for col in (set(defaults) - column_set):
                    accum.append(col)

                columns = sorted(accum, key=lambda obj: obj.get_sort_key(ctx))
            rows_iter = itertools.chain(iter((row,)), rows_iter)
        else:
            clean_columns = []
            seen = set()
            for column in columns:
                if isinstance(column, basestring):
                    column_obj = getattr(self.table, column)
                else:
                    column_obj = column
                clean_columns.append(column_obj)
                seen.add(column_obj)

            columns = clean_columns
            for col in sorted(defaults, key=lambda obj: obj.get_sort_key(ctx)):
                if col not in seen:
                    columns.append(col)

        value_lookups = {}
        for column in columns:
            lookups = [column, column.name]
            if isinstance(column, Field) and column.name != column.column_name:
                lookups.append(column.column_name)
            value_lookups[column] = lookups

        ctx.sql(EnclosedNodeList(columns)).literal(' VALUES ')
        columns_converters = [
            (column, column.db_value if isinstance(column, Field) else None)
            for column in columns]

        all_values = []
        for row in rows_iter:
            values = []
            is_dict = isinstance(row, Mapping)
            for i, (column, converter) in enumerate(columns_converters):
                try:
                    if is_dict:
                        # The logic is a bit convoluted, but in order to be
                        # flexible in what we accept (dict keyed by
                        # column/field, field name, or underlying column name),
                        # we try accessing the row data dict using each
                        # possible key. If no match is found, throw an error.
                        for lookup in value_lookups[column]:
                            try:
                                val = row[lookup]
                            except KeyError:
                                pass
                            else:
                                break
                        else:
                            raise KeyError
                    else:
                        val = row[i]
                except (KeyError, IndexError):
                    if column in defaults:
                        val = defaults[column]
                        if callable_(val):
                            val = val()
                    else:
                        raise ValueError('Missing value for %s.' % column.name)

                if not isinstance(val, Node):
                    val = Value(val, converter=converter, unpack=False)
                values.append(val)

            all_values.append(EnclosedNodeList(values))

        if not all_values:
            raise self.DefaultValuesException('Error: no data to insert.')

        with ctx.scope_values(subquery=True):
            return ctx.sql(CommaNodeList(all_values))

    def _query_insert(self, ctx):
        return (ctx
                .sql(EnclosedNodeList(self._columns))
                .literal(' ')
                .sql(self._insert))

    def _default_values(self, ctx):
        if not self._context:
            return ctx.literal('DEFAULT VALUES')
        return self._context.default_values_insert(ctx)

    def __sql__(self, ctx):
        super(Insert, self).__sql__(ctx)
        with ctx.scope_values():
            stmt = None
            if self._on_conflict is not None:
                stmt = self._on_conflict.get_conflict_statement(ctx, self)

            (ctx
             .sql(stmt or SQL('INSERT'))
             .literal(' INTO ')
             .sql(self.table)
             .literal(' '))

            if isinstance(self._insert, Mapping) and not self._columns:
                try:
                    self._simple_insert(ctx)
                except self.DefaultValuesException:
                    self._default_values(ctx)
                self._query_type = Insert.SIMPLE
            elif isinstance(self._insert, (SelectQuery, SQL)):
                self._query_insert(ctx)
                self._query_type = Insert.QUERY
            else:
                self._generate_insert(self._insert, ctx)
                self._query_type = Insert.MULTI

            if self._on_conflict is not None:
                update = self._on_conflict.get_conflict_update(ctx, self)
                if update is not None:
                    ctx.literal(' ').sql(update)

            return self.apply_returning(ctx)


class Delete(_WriteQuery):
    def __sql__(self, ctx):
        super(Delete, self).__sql__(ctx)

        with ctx.scope_values(subquery=True):
            ctx.literal('DELETE FROM ').sql(self.table)
            if self._where is not None:
                with ctx.scope_normal():
                    ctx.literal(' WHERE ').sql(self._where)

            self._apply_ordering(ctx)
            return self.apply_returning(ctx)


class Index(Node):
    def __init__(self, name, table, expressions, unique=False, safe=False,
                 where=None, using=None):
        self._name = name
        self._table = Entity(table) if not isinstance(table, Table) else table
        self._expressions = expressions
        self._where = where
        self._unique = unique
        self._safe = safe
        self._using = using

    @Node.copy
    def safe(self, _safe=True):
        self._safe = _safe

    @Node.copy
    def where(self, *expressions):
        if self._where is not None:
            expressions = (self._where,) + expressions
        self._where = reduce(operator.and_, expressions)

    @Node.copy
    def using(self, _using=None):
        self._using = _using

    def __sql__(self, ctx):
        statement = 'CREATE UNIQUE INDEX ' if self._unique else 'CREATE INDEX '
        with ctx.scope_values(subquery=True):
            ctx.literal(statement)
            if self._safe:
                ctx.literal('IF NOT EXISTS ')

            # Sqlite uses CREATE INDEX <schema>.<name> ON <table>, whereas most
            # others use: CREATE INDEX <name> ON <schema>.<table>.
            if ctx.state.index_schema_prefix and \
                    isinstance(self._table, Table) and self._table._schema:
                index_name = Entity(self._table._schema, self._name)
                table_name = Entity(self._table.__name__)
            else:
                index_name = Entity(self._name)
                table_name = self._table

            ctx.sql(index_name)
            if self._using is not None and \
                    ctx.state.index_using_precedes_table:
                ctx.literal(' USING %s' % self._using)  # MySQL style.

            (ctx
             .literal(' ON ')
             .sql(table_name)
             .literal(' '))

            if self._using is not None and not \
                    ctx.state.index_using_precedes_table:
                ctx.literal('USING %s ' % self._using)  # Postgres/default.

            ctx.sql(EnclosedNodeList([
                SQL(expr) if isinstance(expr, basestring) else expr
                for expr in self._expressions]))
            if self._where is not None:
                ctx.literal(' WHERE ').sql(self._where)

        return ctx


class ModelIndex(Index):
    def __init__(self, model, fields, unique=False, safe=True, where=None,
                 using=None, name=None):
        self._model = model
        if name is None:
            name = self._generate_name_from_fields(model, fields)
        if using is None:
            for field in fields:
                if isinstance(field, Field) and hasattr(field, 'index_type'):
                    using = field.index_type
        super(ModelIndex, self).__init__(
            name=name,
            table=model._meta.table,
            expressions=fields,
            unique=unique,
            safe=safe,
            where=where,
            using=using)

    def _generate_name_from_fields(self, model, fields):
        accum = []
        for field in fields:
            if isinstance(field, basestring):
                accum.append(field.split()[0])
            else:
                if isinstance(field, Node) and not isinstance(field, Field):
                    field = field.unwrap()
                if isinstance(field, Field):
                    accum.append(field.column_name)

        if not accum:
            raise ValueError('Unable to generate a name for the index, please '
                             'explicitly specify a name.')

        clean_field_names = re.sub(r'[^\w]+', '', '_'.join(accum))
        meta = model._meta
        prefix = meta.name if meta.legacy_table_names else meta.table_name
        return _truncate_constraint_name('_'.join((prefix, clean_field_names)))


def _truncate_constraint_name(constraint, maxlen=64):
    if len(constraint) > maxlen:
        name_hash = hashlib.md5(constraint.encode('utf-8')).hexdigest()
        constraint = '%s_%s' % (constraint[:(maxlen - 8)], name_hash[:7])
    return constraint


# FIELDS

class FieldAccessor(object):
    def __init__(self, model, field, name):
        self.model = model
        self.field = field
        self.name = name

    def __get__(self, instance, instance_type=None):
        if instance is not None:
            return instance.__data__.get(self.name)
        return self.field

    def __set__(self, instance, value):
        instance.__data__[self.name] = value
        instance._dirty.add(self.name)


class ForeignKeyAccessor(FieldAccessor):
    def __init__(self, model, field, name):
        super(ForeignKeyAccessor, self).__init__(model, field, name)
        self.rel_model = field.rel_model

    def get_rel_instance(self, instance):
        value = instance.__data__.get(self.name)
        if value is not None or self.name in instance.__rel__:
            if self.name not in instance.__rel__:
                obj = self.rel_model.get(self.field.rel_field == value)
                instance.__rel__[self.name] = obj
            return instance.__rel__[self.name]
        elif not self.field.null:
            raise self.rel_model.DoesNotExist
        return value

    def __get__(self, instance, instance_type=None):
        if instance is not None:
            return self.get_rel_instance(instance)
        return self.field

    def __set__(self, instance, obj):
        if isinstance(obj, self.rel_model):
            instance.__data__[self.name] = getattr(obj, self.field.rel_field.name)
            instance.__rel__[self.name] = obj
        else:
            fk_value = instance.__data__.get(self.name)
            instance.__data__[self.name] = obj
            if obj != fk_value and self.name in instance.__rel__:
                del instance.__rel__[self.name]
        instance._dirty.add(self.name)


class NoQueryForeignKeyAccessor(ForeignKeyAccessor):
    def get_rel_instance(self, instance):
        value = instance.__data__.get(self.name)
        if value is not None:
            return instance.__rel__.get(self.name, value)
        elif not self.field.null:
            raise self.rel_model.DoesNotExist


class BackrefAccessor(object):
    def __init__(self, field):
        self.field = field
        self.model = field.rel_model
        self.rel_model = field.model

    def __get__(self, instance, instance_type=None):
        if instance is not None:
            dest = self.field.rel_field.name
            return (self.rel_model
                    .select()
                    .where(self.field == getattr(instance, dest)))
        return self


class ObjectIdAccessor(object):
    """Gives direct access to the underlying id"""

    def __init__(self, field):
        self.field = field

    def __get__(self, instance, instance_type=None):
        if instance is not None:
            value = instance.__data__.get(self.field.name)
            # Pull the object-id from the related object if it is not set.
            if value is None and self.field.name in instance.__rel__:
                rel_obj = instance.__rel__[self.field.name]
                value = getattr(rel_obj, self.field.rel_field.name)
            return value
        return self.field

    def __set__(self, instance, value):
        setattr(instance, self.field.name, value)


class Field(ColumnBase):
    _field_counter = 0
    _order = 0
    accessor_class = FieldAccessor
    auto_increment = False
    default_index_type = None
    field_type = 'DEFAULT'
    unpack = True

    def __init__(self, null=False, index=False, unique=False, column_name=None,
                 default=None, primary_key=False, constraints=None,
                 sequence=None, collation=None, unindexed=False, choices=None,
                 help_text=None, verbose_name=None, index_type=None,
                 db_column=None, _hidden=False):
        if db_column is not None:
            __deprecated__('"db_column" has been deprecated in favor of '
                           '"column_name" for Field objects.')
            column_name = db_column

        self.null = null
        self.index = index
        self.unique = unique
        self.column_name = column_name
        self.default = default
        self.primary_key = primary_key
        self.constraints = constraints  # List of column constraints.
        self.sequence = sequence  # Name of sequence, e.g. foo_id_seq.
        self.collation = collation
        self.unindexed = unindexed
        self.choices = choices
        self.help_text = help_text
        self.verbose_name = verbose_name
        self.index_type = index_type or self.default_index_type
        self._hidden = _hidden

        # Used internally for recovering the order in which Fields were defined
        # on the Model class.
        Field._field_counter += 1
        self._order = Field._field_counter
        self._sort_key = (self.primary_key and 1 or 2), self._order

    def __hash__(self):
        return hash(self.name + '.' + self.model.__name__)

    def __repr__(self):
        if hasattr(self, 'model') and getattr(self, 'name', None):
            return '<%s: %s.%s>' % (type(self).__name__,
                                    self.model.__name__,
                                    self.name)
        return '<%s: (unbound)>' % type(self).__name__

    def bind(self, model, name, set_attribute=True):
        self.model = model
        self.name = self.safe_name = name
        self.column_name = self.column_name or name
        if set_attribute:
            setattr(model, name, self.accessor_class(model, self, name))

    @property
    def column(self):
        return Column(self.model._meta.table, self.column_name)

    def adapt(self, value):
        return value

    def db_value(self, value):
        return value if value is None else self.adapt(value)

    def python_value(self, value):
        return value if value is None else self.adapt(value)

    def to_value(self, value):
        return Value(value, self.db_value, unpack=False)

    def get_sort_key(self, ctx):
        return self._sort_key

    def __sql__(self, ctx):
        return ctx.sql(self.column)

    def get_modifiers(self):
        pass

    def ddl_datatype(self, ctx):
        if ctx and ctx.state.field_types:
            column_type = ctx.state.field_types.get(self.field_type,
                                                    self.field_type)
        else:
            column_type = self.field_type

        modifiers = self.get_modifiers()
        if column_type and modifiers:
            modifier_literal = ', '.join([str(m) for m in modifiers])
            return SQL('%s(%s)' % (column_type, modifier_literal))
        else:
            return SQL(column_type)

    def ddl(self, ctx):
        accum = [Entity(self.column_name)]
        data_type = self.ddl_datatype(ctx)
        if data_type:
            accum.append(data_type)
        if self.unindexed:
            accum.append(SQL('UNINDEXED'))
        if not self.null:
            accum.append(SQL('NOT NULL'))
        if self.primary_key:
            accum.append(SQL('PRIMARY KEY'))
        if self.sequence:
            accum.append(SQL("DEFAULT NEXTVAL('%s')" % self.sequence))
        if self.constraints:
            accum.extend(self.constraints)
        if self.collation:
            accum.append(SQL('COLLATE %s' % self.collation))
        return NodeList(accum)


class IntegerField(Field):
    field_type = 'INT'

    def adapt(self, value):
        try:
            return int(value)
        except ValueError:
            return value


class BigIntegerField(IntegerField):
    field_type = 'BIGINT'


class SmallIntegerField(IntegerField):
    field_type = 'SMALLINT'


class AutoField(IntegerField):
    auto_increment = True
    field_type = 'AUTO'

    def __init__(self, *args, **kwargs):
        if kwargs.get('primary_key') is False:
            raise ValueError('%s must always be a primary key.' % type(self))
        kwargs['primary_key'] = True
        super(AutoField, self).__init__(*args, **kwargs)


class BigAutoField(AutoField):
    field_type = 'BIGAUTO'


class IdentityField(AutoField):
    field_type = 'INT GENERATED BY DEFAULT AS IDENTITY'

    def __init__(self, generate_always=False, **kwargs):
        if generate_always:
            self.field_type = 'INT GENERATED ALWAYS AS IDENTITY'
        super(IdentityField, self).__init__(**kwargs)


class FloatField(Field):
    field_type = 'FLOAT'

    def adapt(self, value):
        try:
            return float(value)
        except ValueError:
            return value


class DoubleField(FloatField):
    field_type = 'DOUBLE'


class DecimalField(Field):
    field_type = 'DECIMAL'

    def __init__(self, max_digits=10, decimal_places=5, auto_round=False,
                 rounding=None, *args, **kwargs):
        self.max_digits = max_digits
        self.decimal_places = decimal_places
        self.auto_round = auto_round
        self.rounding = rounding or decimal.DefaultContext.rounding
        self._exp = decimal.Decimal(10) ** (-self.decimal_places)
        super(DecimalField, self).__init__(*args, **kwargs)

    def get_modifiers(self):
        return [self.max_digits, self.decimal_places]

    def db_value(self, value):
        D = decimal.Decimal
        if not value:
            return value if value is None else D(0)
        if self.auto_round:
            decimal_value = D(text_type(value))
            return decimal_value.quantize(self._exp, rounding=self.rounding)
        return value

    def python_value(self, value):
        if value is not None:
            if isinstance(value, decimal.Decimal):
                return value
            return decimal.Decimal(text_type(value))


class _StringField(Field):
    def adapt(self, value):
        if isinstance(value, text_type):
            return value
        elif isinstance(value, bytes_type):
            return value.decode('utf-8')
        return text_type(value)

    def __add__(self, other):
        return StringExpression(self, OP.CONCAT, other)

    def __radd__(self, other):
        return StringExpression(other, OP.CONCAT, self)


class CharField(_StringField):
    field_type = 'VARCHAR'

    def __init__(self, max_length=255, *args, **kwargs):
        self.max_length = max_length
        super(CharField, self).__init__(*args, **kwargs)

    def get_modifiers(self):
        return self.max_length and [self.max_length] or None


class FixedCharField(CharField):
    field_type = 'CHAR'

    def python_value(self, value):
        value = super(FixedCharField, self).python_value(value)
        if value:
            value = value.strip()
        return value


class TextField(_StringField):
    field_type = 'TEXT'


class BlobField(Field):
    field_type = 'BLOB'
    _constructor = bytes

    def db_value(self, value):
        if isinstance(value, text_type):
            value = value.encode('raw_unicode_escape')
        if isinstance(value, bytes_type):
            return self._constructor(value)
        return value


class BitField(BitwiseMixin, BigIntegerField):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault('default', 0)
        super(BitField, self).__init__(*args, **kwargs)
        self.__current_flag = 1

    def flag(self, value=None):
        if value is None:
            value = self.__current_flag
            self.__current_flag <<= 1
        else:
            self.__current_flag = value << 1

        class FlagDescriptor(ColumnBase):
            def __init__(self, field, value):
                self._field = field
                self._value = value
                super(FlagDescriptor, self).__init__()

            def clear(self):
                return self._field.bin_and(~self._value)

            def set(self):
                return self._field.bin_or(self._value)

            def __get__(self, instance, instance_type=None):
                if instance is None:
                    return self
                value = getattr(instance, self._field.name) or 0
                return (value & self._value) != 0

            def __set__(self, instance, is_set):
                if is_set not in (True, False):
                    raise ValueError('Value must be either True or False')
                value = getattr(instance, self._field.name) or 0
                if is_set:
                    value |= self._value
                else:
                    value &= ~self._value
                setattr(instance, self._field.name, value)

            def __sql__(self, ctx):
                return ctx.sql(self._field.bin_and(self._value) != 0)

        return FlagDescriptor(self, value)


class BigBitFieldData(object):
    def __init__(self, instance, name):
        self.instance = instance
        self.name = name
        value = self.instance.__data__.get(self.name)
        if not value:
            value = bytearray()
        elif not isinstance(value, bytearray):
            value = bytearray(value)
        self._buffer = self.instance.__data__[self.name] = value

    def _ensure_length(self, idx):
        byte_num, byte_offset = divmod(idx, 8)
        cur_size = len(self._buffer)
        if cur_size <= byte_num:
            self._buffer.extend(b'\x00' * ((byte_num + 1) - cur_size))
        return byte_num, byte_offset

    def set_bit(self, idx):
        byte_num, byte_offset = self._ensure_length(idx)
        self._buffer[byte_num] |= (1 << byte_offset)

    def clear_bit(self, idx):
        byte_num, byte_offset = self._ensure_length(idx)
        self._buffer[byte_num] &= ~(1 << byte_offset)

    def toggle_bit(self, idx):
        byte_num, byte_offset = self._ensure_length(idx)
        self._buffer[byte_num] ^= (1 << byte_offset)
        return bool(self._buffer[byte_num] & (1 << byte_offset))

    def is_set(self, idx):
        byte_num, byte_offset = self._ensure_length(idx)
        return bool(self._buffer[byte_num] & (1 << byte_offset))

    def __repr__(self):
        return repr(self._buffer)


class BigBitFieldAccessor(FieldAccessor):
    def __get__(self, instance, instance_type=None):
        if instance is None:
            return self.field
        return BigBitFieldData(instance, self.name)

    def __set__(self, instance, value):
        if isinstance(value, memoryview):
            value = value.tobytes()
        elif isinstance(value, buffer_type):
            value = bytes(value)
        elif isinstance(value, bytearray):
            value = bytes_type(value)
        elif isinstance(value, BigBitFieldData):
            value = bytes_type(value._buffer)
        elif isinstance(value, text_type):
            value = value.encode('utf-8')
        elif not isinstance(value, bytes_type):
            raise ValueError('Value must be either a bytes, memoryview or '
                             'BigBitFieldData instance.')
        super(BigBitFieldAccessor, self).__set__(instance, value)


class BigBitField(BlobField):
    accessor_class = BigBitFieldAccessor

    def __init__(self, *args, **kwargs):
        kwargs.setdefault('default', bytes_type)
        super(BigBitField, self).__init__(*args, **kwargs)

    def db_value(self, value):
        return bytes_type(value) if value is not None else value


class UUIDField(Field):
    field_type = 'UUID'

    def db_value(self, value):
        if isinstance(value, basestring) and len(value) == 32:
            # Hex string. No transformation is necessary.
            return value
        elif isinstance(value, bytes) and len(value) == 16:
            # Allow raw binary representation.
            value = uuid.UUID(bytes=value)
        if isinstance(value, uuid.UUID):
            return value.hex
        try:
            return uuid.UUID(value).hex
        except:
            return value

    def python_value(self, value):
        if isinstance(value, uuid.UUID):
            return value
        return uuid.UUID(value) if value is not None else None


class BinaryUUIDField(BlobField):
    field_type = 'UUIDB'

    def db_value(self, value):
        if isinstance(value, bytes) and len(value) == 16:
            # Raw binary value. No transformation is necessary.
            return self._constructor(value)
        elif isinstance(value, basestring) and len(value) == 32:
            # Allow hex string representation.
            value = uuid.UUID(hex=value)
        if isinstance(value, uuid.UUID):
            return self._constructor(value.bytes)
        elif value is not None:
            raise ValueError('value for binary UUID field must be UUID(), '
                             'a hexadecimal string, or a bytes object.')

    def python_value(self, value):
        if isinstance(value, uuid.UUID):
            return value
        elif isinstance(value, memoryview):
            value = value.tobytes()
        elif value and not isinstance(value, bytes):
            value = bytes(value)
        return uuid.UUID(bytes=value) if value is not None else None


def _date_part(date_part):
    def dec(self):
        return self.model._meta.context.extract_date(date_part, self)

    return dec


def format_date_time(value, formats, post_process=None):
    post_process = post_process or (lambda x: x)
    for fmt in formats:
        try:
            return post_process(datetime.datetime.strptime(value, fmt))
        except ValueError:
            pass
    return value


def simple_date_time(value):
    try:
        return datetime.datetime.strptime(value, '%Y-%m-%d %H:%M:%S')
    except (TypeError, ValueError):
        return value


class _BaseFormattedField(Field):
    formats = None

    def __init__(self, formats=None, *args, **kwargs):
        if formats is not None:
            self.formats = formats
        super(_BaseFormattedField, self).__init__(*args, **kwargs)


class DateTimeField(_BaseFormattedField):
    field_type = 'DATETIME'
    formats = [
        '%Y-%m-%d %H:%M:%S.%f',
        '%Y-%m-%d %H:%M:%S',
        '%Y-%m-%d',
    ]

    def adapt(self, value):
        if value and isinstance(value, basestring):
            return format_date_time(value, self.formats)
        return value

    def to_timestamp(self):
        return self.model._meta.context.to_timestamp(self)

    def truncate(self, part):
        return self.model._meta.context.truncate_date(part, self)

    year = property(_date_part('year'))
    month = property(_date_part('month'))
    day = property(_date_part('day'))
    hour = property(_date_part('hour'))
    minute = property(_date_part('minute'))
    second = property(_date_part('second'))


class DateField(_BaseFormattedField):
    field_type = 'DATE'
    formats = [
        '%Y-%m-%d',
        '%Y-%m-%d %H:%M:%S',
        '%Y-%m-%d %H:%M:%S.%f',
    ]

    def adapt(self, value):
        if value and isinstance(value, basestring):
            pp = lambda x: x.date()
            return format_date_time(value, self.formats, pp)
        elif value and isinstance(value, datetime.datetime):
            return value.date()
        return value

    def to_timestamp(self):
        return self.model._meta.context.to_timestamp(self)

    def truncate(self, part):
        return self.model._meta.context.truncate_date(part, self)

    year = property(_date_part('year'))
    month = property(_date_part('month'))
    day = property(_date_part('day'))


class TimeField(_BaseFormattedField):
    field_type = 'TIME'
    formats = [
        '%H:%M:%S.%f',
        '%H:%M:%S',
        '%H:%M',
        '%Y-%m-%d %H:%M:%S.%f',
        '%Y-%m-%d %H:%M:%S',
    ]

    def adapt(self, value):
        if value:
            if isinstance(value, basestring):
                pp = lambda x: x.time()
                return format_date_time(value, self.formats, pp)
            elif isinstance(value, datetime.datetime):
                return value.time()
        if value is not None and isinstance(value, datetime.timedelta):
            return (datetime.datetime.min + value).time()
        return value

    hour = property(_date_part('hour'))
    minute = property(_date_part('minute'))
    second = property(_date_part('second'))


def _timestamp_date_part(date_part):
    def dec(self):
        context = self.model._meta.context
        expr = ((self / Value(self.resolution, converter=False))
                if self.resolution > 1 else self)
        return context.extract_date(date_part, context.from_timestamp(expr))

    return dec


class TimestampField(BigIntegerField):
    # Support second -> microsecond resolution.
    valid_resolutions = [10 ** i for i in range(7)]

    def __init__(self, *args, **kwargs):
        self.resolution = kwargs.pop('resolution', None)

        if not self.resolution:
            self.resolution = 1
        elif self.resolution in range(2, 7):
            self.resolution = 10 ** self.resolution
        elif self.resolution not in self.valid_resolutions:
            raise ValueError('TimestampField resolution must be one of: %s' %
                             ', '.join(str(i) for i in self.valid_resolutions))
        self.ticks_to_microsecond = 1000000 // self.resolution

        self.utc = kwargs.pop('utc', False) or False
        dflt = datetime.datetime.utcnow if self.utc else datetime.datetime.now
        kwargs.setdefault('default', dflt)
        super(TimestampField, self).__init__(*args, **kwargs)

    def local_to_utc(self, dt):
        # Convert naive local datetime into naive UTC, e.g.:
        # 2019-03-01T12:00:00 (local=US/Central) -> 2019-03-01T18:00:00.
        # 2019-05-01T12:00:00 (local=US/Central) -> 2019-05-01T17:00:00.
        # 2019-03-01T12:00:00 (local=UTC)        -> 2019-03-01T12:00:00.
        return datetime.datetime(*time.gmtime(time.mktime(dt.timetuple()))[:6])

    def utc_to_local(self, dt):
        # Convert a naive UTC datetime into local time, e.g.:
        # 2019-03-01T18:00:00 (local=US/Central) -> 2019-03-01T12:00:00.
        # 2019-05-01T17:00:00 (local=US/Central) -> 2019-05-01T12:00:00.
        # 2019-03-01T12:00:00 (local=UTC)        -> 2019-03-01T12:00:00.
        ts = calendar.timegm(dt.utctimetuple())
        return datetime.datetime.fromtimestamp(ts)

    def get_timestamp(self, value):
        if self.utc:
            # If utc-mode is on, then we assume all naive datetimes are in UTC.
            return calendar.timegm(value.utctimetuple())
        else:
            return time.mktime(value.timetuple())

    def db_value(self, value):
        if value is None:
            return

        if isinstance(value, datetime.datetime):
            pass
        elif isinstance(value, datetime.date):
            value = datetime.datetime(value.year, value.month, value.day)
        else:
            return int(round(value * self.resolution))

        timestamp = self.get_timestamp(value)
        if self.resolution > 1:
            timestamp += (value.microsecond * .000001)
            timestamp *= self.resolution
        return int(round(timestamp))

    def python_value(self, value):
        if value is not None and isinstance(value, (int, float, long)):
            if self.resolution > 1:
                value, ticks = divmod(value, self.resolution)
                microseconds = int(ticks * self.ticks_to_microsecond)
            else:
                microseconds = 0

            if self.utc:
                value = datetime.datetime.utcfromtimestamp(value)
            else:
                value = datetime.datetime.fromtimestamp(value)

            if microseconds:
                value = value.replace(microsecond=microseconds)

        return value

    def from_timestamp(self):
        expr = ((self / Value(self.resolution, converter=False))
                if self.resolution > 1 else self)
        return self.model._meta.context.from_timestamp(expr)

    year = property(_timestamp_date_part('year'))
    month = property(_timestamp_date_part('month'))
    day = property(_timestamp_date_part('day'))
    hour = property(_timestamp_date_part('hour'))
    minute = property(_timestamp_date_part('minute'))
    second = property(_timestamp_date_part('second'))


class MetaField(Field):
    column_name = default = model = name = None
    primary_key = False


class IPField(BigIntegerField):
    def db_value(self, val):
        if val is not None:
            return struct.unpack('!I', socket.inet_aton(val))[0]

    def python_value(self, val):
        if val is not None:
            return socket.inet_ntoa(struct.pack('!I', val))


class BooleanField(Field):
    field_type = 'BOOL'
    adapt = bool


class BareField(Field):
    def __init__(self, adapt=None, *args, **kwargs):
        super(BareField, self).__init__(*args, **kwargs)
        if adapt is not None:
            self.adapt = adapt

    def ddl_datatype(self, ctx):
        return


class ForeignKeyField(Field):
    accessor_class = ForeignKeyAccessor

    def __init__(self, model, field=None, backref=None, on_delete=None,
                 on_update=None, deferrable=None, _deferred=None,
                 rel_model=None, to_field=None, object_id_name=None,
                 lazy_load=True, related_name=None, *args, **kwargs):
        kwargs.setdefault('index', True)

        # If lazy_load is disable, we use a different descriptor/accessor that
        # will ensure we don't accidentally perform a query.
        if not lazy_load:
            self.accessor_class = NoQueryForeignKeyAccessor

        super(ForeignKeyField, self).__init__(*args, **kwargs)

        if rel_model is not None:
            __deprecated__('"rel_model" has been deprecated in favor of '
                           '"model" for ForeignKeyField objects.')
            model = rel_model
        if to_field is not None:
            __deprecated__('"to_field" has been deprecated in favor of '
                           '"field" for ForeignKeyField objects.')
            field = to_field
        if related_name is not None:
            __deprecated__('"related_name" has been deprecated in favor of '
                           '"backref" for Field objects.')
            backref = related_name

        self._is_self_reference = model == 'self'
        self.rel_model = model
        self.rel_field = field
        self.declared_backref = backref
        self.backref = None
        self.on_delete = on_delete
        self.on_update = on_update
        self.deferrable = deferrable
        self.deferred = _deferred
        self.object_id_name = object_id_name
        self.lazy_load = lazy_load

    @property
    def field_type(self):
        if not isinstance(self.rel_field, AutoField):
            return self.rel_field.field_type
        elif isinstance(self.rel_field, BigAutoField):
            return BigIntegerField.field_type
        return IntegerField.field_type

    def get_modifiers(self):
        if not isinstance(self.rel_field, AutoField):
            return self.rel_field.get_modifiers()
        return super(ForeignKeyField, self).get_modifiers()

    def adapt(self, value):
        return self.rel_field.adapt(value)

    def db_value(self, value):
        if isinstance(value, self.rel_model):
            value = getattr(value, self.rel_field.name)
        return self.rel_field.db_value(value)

    def python_value(self, value):
        if isinstance(value, self.rel_model):
            return value
        return self.rel_field.python_value(value)

    def bind(self, model, name, set_attribute=True):
        if not self.column_name:
            self.column_name = name if name.endswith('_id') else name + '_id'
        if not self.object_id_name:
            self.object_id_name = self.column_name
            if self.object_id_name == name:
                self.object_id_name += '_id'
        elif self.object_id_name == name:
            raise ValueError('ForeignKeyField "%s"."%s" specifies an '
                             'object_id_name that conflicts with its field '
                             'name.' % (model._meta.name, name))
        if self._is_self_reference:
            self.rel_model = model
        if isinstance(self.rel_field, basestring):
            self.rel_field = getattr(self.rel_model, self.rel_field)
        elif self.rel_field is None:
            self.rel_field = self.rel_model._meta.primary_key

        # Bind field before assigning backref, so field is bound when
        # calling declared_backref() (if callable).
        super(ForeignKeyField, self).bind(model, name, set_attribute)
        self.safe_name = self.object_id_name

        if callable_(self.declared_backref):
            self.backref = self.declared_backref(self)
        else:
            self.backref, self.declared_backref = self.declared_backref, None
        if not self.backref:
            self.backref = '%s_set' % model._meta.name

        if set_attribute:
            setattr(model, self.object_id_name, ObjectIdAccessor(self))
            if self.backref not in '!+':
                setattr(self.rel_model, self.backref, BackrefAccessor(self))

    def foreign_key_constraint(self):
        parts = [
            SQL('FOREIGN KEY'),
            EnclosedNodeList((self,)),
            SQL('REFERENCES'),
            self.rel_model,
            EnclosedNodeList((self.rel_field,))]
        if self.on_delete:
            parts.append(SQL('ON DELETE %s' % self.on_delete))
        if self.on_update:
            parts.append(SQL('ON UPDATE %s' % self.on_update))
        if self.deferrable:
            parts.append(SQL('DEFERRABLE %s' % self.deferrable))
        return NodeList(parts)

    def __getattr__(self, attr):
        if attr.startswith('__'):
            # Prevent recursion error when deep-copying.
            raise AttributeError('Cannot look-up non-existant "__" methods.')
        if attr in self.rel_model._meta.fields:
            return self.rel_model._meta.fields[attr]
        raise AttributeError('Foreign-key has no attribute %s, nor is it a '
                             'valid field on the related model.' % attr)


class CompositeKey(MetaField):
    sequence = None

    def __init__(self, *field_names):
        self.field_names = field_names
        self._safe_field_names = None

    @property
    def safe_field_names(self):
        if self._safe_field_names is None:
            if self.model is None:
                return self.field_names

            self._safe_field_names = [self.model._meta.fields[f].safe_name
                                      for f in self.field_names]
        return self._safe_field_names

    def __get__(self, instance, instance_type=None):
        if instance is not None:
            return tuple([getattr(instance, f) for f in self.safe_field_names])
        return self

    def __set__(self, instance, value):
        if not isinstance(value, (list, tuple)):
            raise TypeError('A list or tuple must be used to set the value of '
                            'a composite primary key.')
        if len(value) != len(self.field_names):
            raise ValueError('The length of the value must equal the number '
                             'of columns of the composite primary key.')
        for idx, field_value in enumerate(value):
            setattr(instance, self.field_names[idx], field_value)

    def __eq__(self, other):
        expressions = [(self.model._meta.fields[field] == value)
                       for field, value in zip(self.field_names, other)]
        return reduce(operator.and_, expressions)

    def __ne__(self, other):
        return ~(self == other)

    def __hash__(self):
        return hash((self.model.__name__, self.field_names))

    def __sql__(self, ctx):
        # If the composite PK is being selected, do not use parens. Elsewhere,
        # such as in an expression, we want to use parentheses and treat it as
        # a row value.
        parens = ctx.scope != SCOPE_SOURCE
        return ctx.sql(NodeList([self.model._meta.fields[field]
                                 for field in self.field_names], ', ', parens))

    def bind(self, model, name, set_attribute=True):
        self.model = model
        self.column_name = self.name = self.safe_name = name
        setattr(model, self.name, self)


class _SortedFieldList(object):
    __slots__ = ('_keys', '_items')

    def __init__(self):
        self._keys = []
        self._items = []

    def __getitem__(self, i):
        return self._items[i]

    def __iter__(self):
        return iter(self._items)

    def __contains__(self, item):
        k = item._sort_key
        i = bisect_left(self._keys, k)
        j = bisect_right(self._keys, k)
        return item in self._items[i:j]

    def index(self, field):
        return self._keys.index(field._sort_key)

    def insert(self, item):
        k = item._sort_key
        i = bisect_left(self._keys, k)
        self._keys.insert(i, k)
        self._items.insert(i, item)

    def remove(self, item):
        idx = self.index(item)
        del self._items[idx]
        del self._keys[idx]


class Metadata(object):
    def __init__(self, model, context=None, table_name=None, indexes=None,
                 primary_key=None, constraints=None, schema=None,
                 only_save_dirty=False, depends_on=None, options=None,
                 db_table=None, table_function=None, table_settings=None,
                 without_rowid=False, temporary=False, legacy_table_names=True,
                 connection_factory=None, slave_connection_factory=None,
                 **kwargs):
        if db_table is not None:
            __deprecated__('"db_table" has been deprecated in favor of '
                           '"table_name" for Models.')
            table_name = db_table
        self.model = model
        self.context = context

        self.fields = {}
        self.columns = {}
        self.combined = {}

        self._sorted_field_list = _SortedFieldList()
        self.sorted_fields = []
        self.sorted_field_names = []

        self.defaults = {}
        self._default_by_name = {}
        self._default_dict = {}
        self._default_callables = {}
        self._default_callable_list = []

        self.name = model.__name__.lower()
        self.table_function = table_function
        self.legacy_table_names = legacy_table_names
        if not table_name:
            table_name = (self.table_function(model)
                          if self.table_function
                          else self.make_table_name())
        self.table_name = table_name
        self._table = None

        self.indexes = list(indexes) if indexes else []
        self.constraints = constraints
        self._schema = schema
        self.primary_key = primary_key
        self.composite_key = self.auto_increment = None
        self.only_save_dirty = only_save_dirty
        self.depends_on = depends_on
        self.table_settings = table_settings
        self.without_rowid = without_rowid
        self.temporary = temporary

        self.refs = {}
        self.backrefs = {}
        self.model_refs = collections.defaultdict(list)
        self.model_backrefs = collections.defaultdict(list)
        self.manytomany = {}

        self.options = options or {}
        for key, value in kwargs.items():
            setattr(self, key, value)
        self._additional_keys = set(kwargs.keys())

        # Allow objects to register hooks that are called if the model is bound
        # to a different database. For example, BlobField uses a different
        # Python data-type depending on the db driver / python version. When
        # the database changes, we need to update any BlobField so they can use
        # the appropriate data-type.
        self._db_hooks = []
        self.connection_factory = connection_factory
        self.slave_connection_factory = slave_connection_factory

    def make_table_name(self):
        if self.legacy_table_names:
            return re.sub(r'[^\w]+', '_', self.name)
        return make_snake_case(self.model.__name__)

    def model_graph(self, refs=True, backrefs=True, depth_first=True):
        if not refs and not backrefs:
            raise ValueError('One of `refs` or `backrefs` must be True.')

        accum = [(None, self.model, None)]
        seen = set()
        queue = collections.deque((self,))
        method = queue.pop if depth_first else queue.popleft

        while queue:
            curr = method()
            if curr in seen: continue
            seen.add(curr)

            if refs:
                for fk, model in curr.refs.items():
                    accum.append((fk, model, False))
                    queue.append(model._meta)
            if backrefs:
                for fk, model in curr.backrefs.items():
                    accum.append((fk, model, True))
                    queue.append(model._meta)

        return accum

    def add_ref(self, field):
        rel = field.rel_model
        self.refs[field] = rel
        self.model_refs[rel].append(field)
        rel._meta.backrefs[field] = self.model
        rel._meta.model_backrefs[self.model].append(field)

    def remove_ref(self, field):
        rel = field.rel_model
        del self.refs[field]
        self.model_refs[rel].remove(field)
        del rel._meta.backrefs[field]
        rel._meta.model_backrefs[self.model].remove(field)

    @property
    def table(self):
        if self._table is None:
            self._table = Table(
                self.table_name,
                [field.column_name for field in self.sorted_fields],
                schema=self.schema,
                _model=self.model,
                _context=self.context)
        return self._table

    @table.setter
    def table(self, value):
        raise AttributeError('Cannot set the "table".')

    @table.deleter
    def table(self):
        self._table = None

    @property
    def schema(self):
        return self._schema

    @schema.setter
    def schema(self, value):
        self._schema = value
        del self.table

    @property
    def entity(self):
        if self._schema:
            return Entity(self._schema, self.table_name)
        else:
            return Entity(self.table_name)

    def _update_sorted_fields(self):
        self.sorted_fields = list(self._sorted_field_list)
        self.sorted_field_names = [f.name for f in self.sorted_fields]

    def get_rel_for_model(self, model):
        if isinstance(model, ModelAlias):
            model = model.model
        forwardrefs = self.model_refs.get(model, [])
        backrefs = self.model_backrefs.get(model, [])
        return (forwardrefs, backrefs)

    def add_field(self, field_name, field, set_attribute=True):
        if field_name in self.fields:
            self.remove_field(field_name)
        if not isinstance(field, MetaField):
            del self.table
            field.bind(self.model, field_name, set_attribute)
            self.fields[field.name] = field
            self.columns[field.column_name] = field
            self.combined[field.name] = field
            self.combined[field.column_name] = field

            self._sorted_field_list.insert(field)
            self._update_sorted_fields()

            if field.default is not None:
                # This optimization helps speed up model instance construction.
                self.defaults[field] = field.default
                if callable_(field.default):
                    self._default_callables[field] = field.default
                    self._default_callable_list.append((field.name,
                                                        field.default))
                else:
                    self._default_dict[field] = field.default
                    self._default_by_name[field.name] = field.default
        else:
            field.bind(self.model, field_name, set_attribute)

        if isinstance(field, ForeignKeyField):
            self.add_ref(field)
        # many to many field is unnecessary
        # elif isinstance(field, ManyToManyField) and field.name:
        #     self.add_manytomany(field)

    def remove_field(self, field_name):
        if field_name not in self.fields:
            return

        del self.table
        original = self.fields.pop(field_name)
        del self.columns[original.column_name]
        del self.combined[field_name]
        try:
            del self.combined[original.column_name]
        except KeyError:
            pass
        self._sorted_field_list.remove(original)
        self._update_sorted_fields()

        if original.default is not None:
            del self.defaults[original]
            if self._default_callables.pop(original, None):
                for i, (name, _) in enumerate(self._default_callable_list):
                    if name == field_name:
                        self._default_callable_list.pop(i)
                        break
            else:
                self._default_dict.pop(original, None)
                self._default_by_name.pop(original.name, None)

        if isinstance(original, ForeignKeyField):
            self.remove_ref(original)

    def set_primary_key(self, name, field):
        self.composite_key = isinstance(field, CompositeKey)
        self.add_field(name, field)
        self.primary_key = field
        self.auto_increment = (
                field.auto_increment or
                bool(field.sequence))

    def get_primary_keys(self):
        if self.composite_key:
            return tuple([self.fields[field_name]
                          for field_name in self.primary_key.field_names])
        else:
            return (self.primary_key,) if self.primary_key is not False else ()

    def get_default_dict(self):
        dd = self._default_by_name.copy()
        for field_name, default in self._default_callable_list:
            dd[field_name] = default()
        return dd

    def fields_to_index(self):
        indexes = []
        for f in self.sorted_fields:
            if f.primary_key:
                continue
            if f.index or f.unique:
                indexes.append(ModelIndex(self.model, (f,), unique=f.unique,
                                          using=f.index_type))

        for index_obj in self.indexes:
            if isinstance(index_obj, Node):
                indexes.append(index_obj)
            elif isinstance(index_obj, (list, tuple)):
                index_parts, unique = index_obj
                fields = []
                for part in index_parts:
                    if isinstance(part, basestring):
                        fields.append(self.combined[part])
                    elif isinstance(part, Node):
                        fields.append(part)
                    else:
                        raise ValueError('Expected either a field name or a '
                                         'subclass of Node. Got: %s' % part)
                indexes.append(ModelIndex(self.model, fields, unique=unique))

        return indexes

    def set_context(self, context):
        self.context = context
        del self.table

        # Apply any hooks that have been registered.
        for hook in self._db_hooks:
            hook(context)

    def set_table_name(self, table_name):
        self.table_name = table_name
        del self.table


class ModelBase(type):
    inheritable = set(['constraints', 'context', 'indexes', 'primary_key',
                       'options', 'schema', 'table_function', 'temporary',
                       'only_save_dirty', 'legacy_table_names', 'connection_factory',
                       'table_settings'])

    def __new__(cls, name, bases, attrs):
        if name == MODEL_BASE or bases[0].__name__ == MODEL_BASE:
            return super(ModelBase, cls).__new__(cls, name, bases, attrs)

        meta_options = {}
        meta = attrs.pop('Meta', None)
        if meta:
            for k, v in meta.__dict__.items():
                if not k.startswith('_'):
                    meta_options[k] = v

        pk = getattr(meta, 'primary_key', None)
        pk_name = parent_pk = None

        # Inherit any field descriptors by deep copying the underlying field
        # into the attrs of the new model, additionally see if the bases define
        # inheritable model options and swipe them.
        for b in bases:
            if not hasattr(b, '_meta'):
                continue

            base_meta = b._meta
            if parent_pk is None:
                parent_pk = deepcopy(base_meta.primary_key)
            all_inheritable = cls.inheritable | base_meta._additional_keys
            for k in base_meta.__dict__:
                if k in all_inheritable and k not in meta_options:
                    meta_options[k] = base_meta.__dict__[k]
            meta_options.setdefault('schema', base_meta.schema)

            for (k, v) in b.__dict__.items():
                if k in attrs: continue

                if isinstance(v, FieldAccessor) and not v.field.primary_key:
                    attrs[k] = deepcopy(v.field)

        Meta = meta_options.get('model_metadata_class', Metadata)
        from .context import MYSQLContext
        meta_options['context'] = MYSQLContext

        # Construct the new class.
        cls = super(ModelBase, cls).__new__(cls, name, bases, attrs)
        cls.__data__ = cls.__rel__ = None

        cls._meta = Meta(cls, **meta_options)

        fields = []
        for key, value in cls.__dict__.items():
            if isinstance(value, Field):
                if value.primary_key and pk:
                    raise ValueError('over-determined primary key %s.' % name)
                elif value.primary_key:
                    pk, pk_name = value, key
                else:
                    fields.append((key, value))

        if pk is None:
            if parent_pk is not False:
                pk, pk_name = ((parent_pk, parent_pk.name)
                               if parent_pk is not None else
                               (AutoField(), 'id'))
            else:
                pk = False
        elif isinstance(pk, CompositeKey):
            pk_name = '__composite_key__'
            cls._meta.composite_key = True

        if pk is not False:
            cls._meta.set_primary_key(pk_name, pk)

        for name, field in fields:
            cls._meta.add_field(name, field)

        # Create a repr and error class before finalizing.
        if hasattr(cls, '__str__') and '__repr__' not in attrs:
            setattr(cls, '__repr__', lambda self: '<%s: %s>' % (
                cls.__name__, self.__str__()))

        exc_name = '%sDoesNotExist' % cls.__name__
        exc_attrs = {'__module__': cls.__module__}
        exception_class = type(exc_name, (DoesNotExist,), exc_attrs)
        cls.DoesNotExist = exception_class

        # Call validation hook, allowing additional model validation.
        cls.validate_model()
        return cls

    def __repr__(self):
        return '<Model: %s>' % self.__name__

    def __bool__(self):
        return True

    __nonzero__ = __bool__  # Python 2.

    def __sql__(self, ctx):
        return ctx.sql(self._meta.table)


class _BoundModelsContext(_callable_context_manager):
    def __init__(self, models, context, bind_refs, bind_backrefs):
        self.models = models
        self.context = context
        self.bind_refs = bind_refs
        self.bind_backrefs = bind_backrefs

    def __enter__(self):
        self._orig_context = []
        for model in self.models:
            self._orig_context.append(model._meta.context)
            model.bind(self.context, self.bind_refs, self.bind_backrefs,
                       _exclude=set(self.models))
        return self.models

    def __exit__(self, exc_type, exc_val, exc_tb):
        for model, context in zip(self.models, self._orig_context):
            model.bind(context, self.bind_refs, self.bind_backrefs,
                       _exclude=set(self.models))


class Model(with_metaclass(ModelBase, Node)):
    def __init__(self, *args, **kwargs):
        if kwargs.pop('__no_default__', None):
            self.__data__ = {}
        else:
            self.__data__ = self._meta.get_default_dict()
        self._dirty = set(self.__data__)
        self.__rel__ = {}

        for k in kwargs:
            setattr(self, k, kwargs[k])

    def __str__(self):
        return str(self._pk) if self._meta.primary_key is not False else 'n/a'

    @classmethod
    def validate_model(cls):
        pass

    @classmethod
    def alias(cls, alias=None):
        return ModelAlias(cls, alias)

    @classmethod
    def select(cls, *fields):
        is_default = not fields
        if not fields:
            fields = cls._meta.sorted_fields
        return ModelSelect(cls, fields, is_default=is_default)

    @classmethod
    def _normalize_data(cls, data, kwargs):
        normalized = {}
        if data:
            if not isinstance(data, dict):
                if kwargs:
                    raise ValueError('Data cannot be mixed with keyword '
                                     'arguments: %s' % data)
                return data
            for key in data:
                try:
                    field = (key if isinstance(key, Field)
                             else cls._meta.combined[key])
                except KeyError:
                    if not isinstance(key, Node):
                        raise ValueError('Unrecognized field name: "%s" in %s.'
                                         % (key, data))
                    field = key
                normalized[field] = data[key]
        if kwargs:
            for key in kwargs:
                try:
                    normalized[cls._meta.combined[key]] = kwargs[key]
                except KeyError:
                    normalized[getattr(cls, key)] = kwargs[key]
        return normalized

    @classmethod
    def update(cls, __data=None, **update):
        return ModelUpdate(cls, cls._normalize_data(__data, update))

    @classmethod
    def insert(cls, __data=None, **insert):
        return ModelInsert(cls, cls._normalize_data(__data, insert))

    @classmethod
    def insert_many(cls, rows, fields=None):
        return ModelInsert(cls, insert=rows, columns=fields)

    @classmethod
    def insert_from(cls, query, fields):
        columns = [getattr(cls, field) if isinstance(field, basestring)
                   else field for field in fields]
        return ModelInsert(cls, insert=query, columns=columns)

    @classmethod
    def replace(cls, __data=None, **insert):
        return cls.insert(__data, **insert).on_conflict('REPLACE')

    @classmethod
    def replace_many(cls, rows, fields=None):
        return (cls
                .insert_many(rows=rows, fields=fields)
                .on_conflict('REPLACE'))

    @classmethod
    def delete(cls):
        return ModelDelete(cls)

    @classmethod
    def create(cls, **query):
        inst = cls(**query)
        inst.save(force_insert=True)
        return inst

    @classmethod
    def noop(cls):
        return NoopModelSelect(cls, ())

    @classmethod
    def get_by_id(cls, pk):
        return cls.get(cls._meta.primary_key == pk)

    @classmethod
    def delete_by_id(cls, pk):
        return cls.delete().where(cls._meta.primary_key == pk)

    @classmethod
    def filter(cls, *dq_nodes, **filters):
        return cls.select().filter(*dq_nodes, **filters)

    def get_id(self):
        # Using getattr(self, pk-name) could accidentally trigger a query if
        # the primary-key is a foreign-key. So we use the safe_name attribute,
        # which defaults to the field-name, but will be the object_id_name for
        # foreign-key fields.
        if self._meta.primary_key is not False:
            return getattr(self, self._meta.primary_key.safe_name)

    _pk = property(get_id)

    @_pk.setter
    def _pk(self, value):
        setattr(self, self._meta.primary_key.name, value)

    def _pk_expr(self):
        return self._meta.primary_key == self._pk

    def _prune_fields(self, field_dict, only):
        new_data = {}
        for field in only:
            if isinstance(field, basestring):
                field = self._meta.combined[field]
            if field.name in field_dict:
                new_data[field.name] = field_dict[field.name]
        return new_data

    def _populate_unsaved_relations(self, field_dict):
        for foreign_key_field in self._meta.refs:
            foreign_key = foreign_key_field.name
            conditions = (
                    foreign_key in field_dict and
                    field_dict[foreign_key] is None and
                    self.__rel__.get(foreign_key) is not None)
            if conditions:
                setattr(self, foreign_key, getattr(self, foreign_key))
                field_dict[foreign_key] = self.__data__[foreign_key]

    def is_dirty(self):
        return bool(self._dirty)

    @property
    def dirty_fields(self):
        return [f for f in self._meta.sorted_fields if f.name in self._dirty]

    def dependencies(self, search_nullable=False):
        model_class = type(self)
        stack = [(type(self), None)]
        seen = set()

        while stack:
            klass, query = stack.pop()
            if klass in seen:
                continue
            seen.add(klass)
            for fk, rel_model in klass._meta.backrefs.items():
                if rel_model is model_class or query is None:
                    node = (fk == self.__data__[fk.rel_field.name])
                else:
                    node = fk << query
                subquery = (rel_model.select(rel_model._meta.primary_key)
                            .where(node))
                if not fk.null or search_nullable:
                    stack.append((rel_model, subquery))
                yield (node, fk)

    def __hash__(self):
        return hash((self.__class__, self._pk))

    def __eq__(self, other):
        return (
                other.__class__ == self.__class__ and
                self._pk is not None and
                self._pk == other._pk)

    def __ne__(self, other):
        return not self == other

    def __sql__(self, ctx):
        return ctx.sql(Value(getattr(self, self._meta.primary_key.name),
                             converter=self._meta.primary_key.db_value))

    @classmethod
    def bind(cls, context, bind_refs=True, bind_backrefs=True, _exclude=None):
        is_different = cls._meta.context is not context
        cls._meta.set_context(context)
        if bind_refs or bind_backrefs:
            if _exclude is None:
                _exclude = set()
            G = cls._meta.model_graph(refs=bind_refs, backrefs=bind_backrefs)
            for _, model, is_backref in G:
                if model not in _exclude:
                    model._meta.set_context(context)
                    _exclude.add(model)
        return is_different

    @classmethod
    def bind_ctx(cls, context, bind_refs=True, bind_backrefs=True):
        return _BoundModelsContext((cls,), context, bind_refs, bind_backrefs)

    @classmethod
    def index(cls, *fields, **kwargs):
        return ModelIndex(cls, fields, **kwargs)

    @classmethod
    def add_index(cls, *fields, **kwargs):
        if len(fields) == 1 and isinstance(fields[0], (SQL, Index)):
            cls._meta.indexes.append(fields[0])
        else:
            cls._meta.indexes.append(ModelIndex(cls, fields, **kwargs))


class ModelAlias(Node):
    """Provide a separate reference to a model in a query."""

    def __init__(self, model, alias=None):
        self.__dict__['model'] = model
        self.__dict__['alias'] = alias

    def __getattr__(self, attr):
        # Hack to work-around the fact that properties or other objects
        # implementing the descriptor protocol (on the model being aliased),
        # will not work correctly when we use getattr(). So we explicitly pass
        # the model alias to the descriptor's getter.
        try:
            obj = self.model.__dict__[attr]
        except KeyError:
            pass
        else:
            if isinstance(obj, ModelDescriptor):
                return obj.__get__(None, self)

        model_attr = getattr(self.model, attr)
        if isinstance(model_attr, Field):
            self.__dict__[attr] = FieldAlias.create(self, model_attr)
            return self.__dict__[attr]
        return model_attr

    def __setattr__(self, attr, value):
        raise AttributeError('Cannot set attributes on model aliases.')

    def get_field_aliases(self):
        return [getattr(self, n) for n in self.model._meta.sorted_field_names]

    def select(self, *selection):
        if not selection:
            selection = self.get_field_aliases()
        return ModelSelect(self, selection)

    def __call__(self, **kwargs):
        return self.model(**kwargs)

    def __sql__(self, ctx):
        if ctx.scope == SCOPE_VALUES:
            # Return the quoted table name.
            return ctx.sql(self.model)

        if self.alias:
            ctx.alias_manager[self] = self.alias

        if ctx.scope == SCOPE_SOURCE:
            # Define the table and its alias.
            return (ctx
                    .sql(self.model._meta.entity)
                    .literal(' AS ')
                    .sql(Entity(ctx.alias_manager[self])))
        else:
            # Refer to the table using the alias.
            return ctx.sql(Entity(ctx.alias_manager[self]))


class FieldAlias(Field):
    def __init__(self, source, field):
        self.source = source
        self.model = source.model
        self.field = field

    @classmethod
    def create(cls, source, field):
        class _FieldAlias(cls, type(field)):
            pass

        return _FieldAlias(source, field)

    def clone(self):
        return FieldAlias(self.source, self.field)

    def adapt(self, value): return self.field.adapt(value)

    def python_value(self, value): return self.field.python_value(value)

    def db_value(self, value): return self.field.db_value(value)

    def __getattr__(self, attr):
        return self.source if attr == 'model' else getattr(self.field, attr)

    def __sql__(self, ctx):
        return ctx.sql(Column(self.source, self.field.column_name))


class _ModelQueryHelper(object):
    default_row_type = ROW.MODEL

    def __init__(self, *args, **kwargs):
        super(_ModelQueryHelper, self).__init__(*args, **kwargs)
        if not self._context:
            self._context = self.model._meta.context

    @Node.copy
    def objects(self, constructor=None):
        self._row_type = ROW.CONSTRUCTOR
        self._constructor = self.model if constructor is None else constructor


class BaseModelSelect(_ModelQueryHelper):
    SINGLE = 0
    MULTI = 1
    SINGLE_VALUE = 2
    MULTI_VALUE = 3

    def union_all(self, rhs):
        return ModelCompoundSelectQuery(self.model, self, 'UNION ALL', rhs)

    __add__ = union_all

    def union(self, rhs):
        return ModelCompoundSelectQuery(self.model, self, 'UNION', rhs)

    __or__ = union

    def intersect(self, rhs):
        return ModelCompoundSelectQuery(self.model, self, 'INTERSECT', rhs)

    __and__ = intersect

    def except_(self, rhs):
        return ModelCompoundSelectQuery(self.model, self, 'EXCEPT', rhs)

    __sub__ = except_

    @Node.copy
    def group_by(self, *columns):
        grouping = []
        for column in columns:
            if is_model(column):
                grouping.extend(column._meta.sorted_fields)
            elif isinstance(column, Table):
                if not column._columns:
                    raise ValueError('Cannot pass a table to group_by() that '
                                     'does not have columns explicitly '
                                     'declared.')
                grouping.extend([getattr(column, col_name)
                                 for col_name in column._columns])
            else:
                grouping.append(column)
        self._group_by = grouping

    def single(self):
        self.query_type = self.SINGLE
        return self

    def many(self):
        self.query_type = self.MULTI
        return self

    def single_value(self):
        self.query_type = self.SINGLE_VALUE
        return self

    def many_value(self):
        self.query_type = self.MULTI_VALUE
        return self

    async def _execute(self, conn=None, context=None):
        if self.query_type == self.SINGLE:
            return await conn.select_one(*self.sql())
        elif self.query_type == self.MULTI:
            return await conn.select_many(*self.sql())
        elif self.query_type == self.SINGLE_VALUE:
            return await conn.select_one_value(*self.sql())
        elif self.query_type == self.MULTI_VALUE:
            return await conn.select_many_one_value(*self.sql())
        else:
            raise RuntimeError("unknown query type")


class ModelCompoundSelectQuery(BaseModelSelect, CompoundSelectQuery):
    def __init__(self, model, *args, **kwargs):
        self.model = model
        super(ModelCompoundSelectQuery, self).__init__(*args, **kwargs)


def _normalize_model_select(fields_or_models):
    fields = []
    for fm in fields_or_models:
        if is_model(fm):
            fields.extend(fm._meta.sorted_fields)
        elif isinstance(fm, ModelAlias):
            fields.extend(fm.get_field_aliases())
        elif isinstance(fm, Table) and fm._columns:
            fields.extend([getattr(fm, col) for col in fm._columns])
        else:
            fields.append(fm)
    return fields


class ModelSelect(BaseModelSelect, Select):

    def __init__(self, model, fields_or_models, is_default=False, t=1):
        self.model = self._join_ctx = model
        self._joins = {}
        self._is_default = is_default
        self.query_type = t
        fields = _normalize_model_select(fields_or_models)
        super(ModelSelect, self).__init__([model], fields)

    def clone(self):
        clone = super(ModelSelect, self).clone()
        if clone._joins:
            clone._joins = dict(clone._joins)
        return clone

    def select(self, *fields_or_models):
        if fields_or_models or not self._is_default:
            self._is_default = False
            fields = _normalize_model_select(fields_or_models)
            return super(ModelSelect, self).select(*fields)
        return self

    def switch(self, ctx=None):
        self._join_ctx = self.model if ctx is None else ctx
        return self

    def _get_model(self, src):
        if is_model(src):
            return src, True
        elif isinstance(src, Table) and src._model:
            return src._model, False
        elif isinstance(src, ModelAlias):
            return src.model, False
        elif isinstance(src, ModelSelect):
            return src.model, False
        return None, False

    def _normalize_join(self, src, dest, on, attr):
        # Allow "on" expression to have an alias that determines the
        # destination attribute for the joined data.
        on_alias = isinstance(on, Alias)
        if on_alias:
            attr = attr or on._alias
            on = on.alias()

        # Obtain references to the source and destination models being joined.
        src_model, src_is_model = self._get_model(src)
        dest_model, dest_is_model = self._get_model(dest)

        if src_model and dest_model:
            self._join_ctx = dest
            constructor = dest_model

            # In the case where the "on" clause is a Column or Field, we will
            # convert that field into the appropriate predicate expression.
            if not (src_is_model and dest_is_model) and isinstance(on, Column):
                if on.source is src:
                    to_field = src_model._meta.columns[on.name]
                elif on.source is dest:
                    to_field = dest_model._meta.columns[on.name]
                else:
                    raise AttributeError('"on" clause Column %s does not '
                                         'belong to %s or %s.' %
                                         (on, src_model, dest_model))
                on = None
            elif isinstance(on, Field):
                to_field = on
                on = None
            else:
                to_field = None

            fk_field, is_backref = self._generate_on_clause(
                src_model, dest_model, to_field, on)

            if on is None:
                src_attr = 'name' if src_is_model else 'column_name'
                dest_attr = 'name' if dest_is_model else 'column_name'
                if is_backref:
                    lhs = getattr(dest, getattr(fk_field, dest_attr))
                    rhs = getattr(src, getattr(fk_field.rel_field, src_attr))
                else:
                    lhs = getattr(src, getattr(fk_field, src_attr))
                    rhs = getattr(dest, getattr(fk_field.rel_field, dest_attr))
                on = (lhs == rhs)

            if not attr:
                if fk_field is not None and not is_backref:
                    attr = fk_field.name
                else:
                    attr = dest_model._meta.name
            elif on_alias and fk_field is not None and \
                    attr == fk_field.object_id_name and not is_backref:
                raise ValueError('Cannot assign join alias to "%s", as this '
                                 'attribute is the object_id_name for the '
                                 'foreign-key field "%s"' % (attr, fk_field))

        elif isinstance(dest, Source):
            constructor = dict
            attr = attr or dest._alias
            if not attr and isinstance(dest, Table):
                attr = attr or dest.__name__

        return (on, attr, constructor)

    def _generate_on_clause(self, src, dest, to_field=None, on=None):
        meta = src._meta
        is_backref = fk_fields = False

        # Get all the foreign keys between source and dest, and determine if
        # the join is via a back-reference.
        if dest in meta.model_refs:
            fk_fields = meta.model_refs[dest]
        elif dest in meta.model_backrefs:
            fk_fields = meta.model_backrefs[dest]
            is_backref = True

        if not fk_fields:
            if on is not None:
                return None, False
            raise ValueError('Unable to find foreign key between %s and %s. '
                             'Please specify an explicit join condition.' %
                             (src, dest))
        elif to_field is not None:
            # If the foreign-key field was specified explicitly, remove all
            # other foreign-key fields from the list.
            target = (to_field.field if isinstance(to_field, FieldAlias)
                      else to_field)
            fk_fields = [f for f in fk_fields if (
                    (f is target) or
                    (is_backref and f.rel_field is to_field))]

        if len(fk_fields) == 1:
            return fk_fields[0], is_backref

        if on is None:
            # If multiple foreign-keys exist, try using the FK whose name
            # matches that of the related model. If not, raise an error as this
            # is ambiguous.
            for fk in fk_fields:
                if fk.name == dest._meta.name:
                    return fk, is_backref

            raise ValueError('More than one foreign key between %s and %s.'
                             ' Please specify which you are joining on.' %
                             (src, dest))

        # If there are multiple foreign-keys to choose from and the join
        # predicate is an expression, we'll try to figure out which
        # foreign-key field we're joining on so that we can assign to the
        # correct attribute when resolving the model graph.
        to_field = None
        if isinstance(on, Expression):
            lhs, rhs = on.lhs, on.rhs
            # Coerce to set() so that we force Python to compare using the
            # object's hash rather than equality test, which returns a
            # false-positive due to overriding __eq__.
            fk_set = set(fk_fields)

            if isinstance(lhs, Field):
                lhs_f = lhs.field if isinstance(lhs, FieldAlias) else lhs
                if lhs_f in fk_set:
                    to_field = lhs_f
            elif isinstance(rhs, Field):
                rhs_f = rhs.field if isinstance(rhs, FieldAlias) else rhs
                if rhs_f in fk_set:
                    to_field = rhs_f

        return to_field, False

    @Node.copy
    def join(self, dest, join_type=JOIN.INNER, on=None, src=None, attr=None):
        src = self._join_ctx if src is None else src

        if join_type == JOIN.LATERAL or join_type == JOIN.LEFT_LATERAL:
            on = True
        elif join_type != JOIN.CROSS:
            on, attr, constructor = self._normalize_join(src, dest, on, attr)
            if attr:
                self._joins.setdefault(src, [])
                self._joins[src].append((dest, attr, constructor, join_type))
        elif on is not None:
            raise ValueError('Cannot specify on clause with cross join.')

        if not self._from_list:
            raise ValueError('No sources to join on.')

        item = self._from_list.pop()
        self._from_list.append(Join(item, dest, join_type, on))

    def join_from(self, src, dest, join_type=JOIN.INNER, on=None, attr=None):
        return self.join(dest, join_type, on, src, attr)

    def ensure_join(self, lm, rm, on=None, **join_kwargs):
        join_ctx = self._join_ctx
        for dest, _, constructor, _ in self._joins.get(lm, []):
            if dest == rm:
                return self
        return self.switch(lm).join(rm, on=on, **join_kwargs).switch(join_ctx)

    def convert_dict_to_node(self, qdict):
        accum = []
        joins = []
        fks = (ForeignKeyField, BackrefAccessor)
        for key, value in sorted(qdict.items()):
            curr = self.model
            if '__' in key and key.rsplit('__', 1)[1] in DJANGO_MAP:
                key, op = key.rsplit('__', 1)
                op = DJANGO_MAP[op]
            elif value is None:
                op = DJANGO_MAP['is']
            else:
                op = DJANGO_MAP['eq']

            if '__' not in key:
                # Handle simplest case. This avoids joining over-eagerly when a
                # direct FK lookup is all that is required.
                model_attr = getattr(curr, key)
            else:
                for piece in key.split('__'):
                    for dest, attr, _, _ in self._joins.get(curr, ()):
                        if attr == piece or (isinstance(dest, ModelAlias) and
                                             dest.alias == piece):
                            curr = dest
                            break
                    else:
                        model_attr = getattr(curr, piece)
                        if value is not None and isinstance(model_attr, fks):
                            curr = model_attr.rel_model
                            joins.append(model_attr)
            accum.append(op(model_attr, value))
        return accum, joins

    def filter(self, *args, **kwargs):
        # normalize args and kwargs into a new expression
        dq_node = ColumnBase()
        if args:
            dq_node &= reduce(operator.and_, [a.clone() for a in args])
        if kwargs:
            dq_node &= DQ(**kwargs)

        # dq_node should now be an Expression, lhs = Node(), rhs = ...
        q = collections.deque([dq_node])
        dq_joins = []
        seen_joins = set()
        while q:
            curr = q.popleft()
            if not isinstance(curr, Expression):
                continue
            for side, piece in (('lhs', curr.lhs), ('rhs', curr.rhs)):
                if isinstance(piece, DQ):
                    query, joins = self.convert_dict_to_node(piece.query)
                    for join in joins:
                        if join not in seen_joins:
                            dq_joins.append(join)
                            seen_joins.add(join)
                    expression = reduce(operator.and_, query)
                    # Apply values from the DQ object.
                    if piece._negated:
                        expression = Negated(expression)
                    # expression._alias = piece._alias
                    setattr(curr, side, expression)
                else:
                    q.append(piece)

        dq_node = dq_node.rhs

        query = self.clone()
        for field in dq_joins:
            if isinstance(field, ForeignKeyField):
                lm, rm = field.model, field.rel_model
                field_obj = field
            elif isinstance(field, BackrefAccessor):
                lm, rm = field.model, field.rel_model
                field_obj = field.field
            query = query.ensure_join(lm, rm, field_obj)
        return query.where(dq_node)

    def __sql_selection__(self, ctx, is_subquery=False):
        if self._is_default and is_subquery and len(self._returning) > 1 and \
                self.model._meta.primary_key is not False:
            return ctx.sql(self.model._meta.primary_key)

        return ctx.sql(CommaNodeList(self._returning))


class NoopModelSelect(ModelSelect):
    def __sql__(self, ctx):
        return self.model._meta.context.get_noop_select(ctx)


class _ModelWriteQueryHelper(_ModelQueryHelper):
    def __init__(self, model, *args, **kwargs):
        self.model = model
        super(_ModelWriteQueryHelper, self).__init__(model, *args, **kwargs)

    def returning(self, *returning):
        accum = []
        for item in returning:
            if is_model(item):
                accum.extend(item._meta.sorted_fields)
            else:
                accum.append(item)
        return super(_ModelWriteQueryHelper, self).returning(*accum)

    def _set_table_alias(self, ctx):
        table = self.model._meta.table
        ctx.alias_manager[table] = table.__name__


class ModelUpdate(_ModelWriteQueryHelper, Update):

    async def _execute(self, conn=None, context=None):
        affected_rows = await conn.update(*self.sql())
        if affected_rows > 1:
            await conn.rollback()
            raise MultiUpdateNotAllowed('[{}]'.format(query_to_string(self)))
        return affected_rows


class ModelInsert(_ModelWriteQueryHelper, Insert):
    default_row_type = ROW.TUPLE

    def __init__(self, *args, **kwargs):
        super(ModelInsert, self).__init__(*args, **kwargs)
        self._returning = self.model._meta.get_primary_keys()

    def returning(self, *returning):
        # By default ModelInsert will yield a `tuple` containing the
        # primary-key of the newly inserted row. But if we are explicitly
        # specifying a returning clause and have not set a row type, we will
        # default to returning model instances instead.
        if returning and self._row_type is None:
            self._row_type = ROW.MODEL
        return super(ModelInsert, self).returning(*returning)

    def get_default_data(self):
        return self.model._meta.defaults

    def get_default_columns(self):
        fields = self.model._meta.sorted_fields
        return fields[1:] if self.model._meta.auto_increment else fields

    async def _execute(self, conn=None, context=None):
        # return last row_id even if insert multi rows
        aid = await conn.insert_one(*self.sql(), return_auto_increament_id=True)
        return aid


class ModelDelete(_ModelWriteQueryHelper, Delete):

    async def _execute(self, conn=None, context=None):
        affected_rows = await conn.delete(*self.sql())
        if affected_rows > 1:
            await conn.rollback()
            error = '[{}]'.format(query_to_string(self))
            raise MultiDeleteNotAllowed(error)
        return affected_rows

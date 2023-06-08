# -*- coding: utf-8 -*-
# @Time: 2023/6/8 17:10
import operator
import re

from .generic import attrdict

__all__ = [
    'SENTINEL',
    'OP',
    'DJANGO_MAP',
    'FIELD',
    'JOIN',
    'ROW',
    'SCOPE_CTE',
    'SCOPE_COLUMN',
    'SCOPE_NORMAL',
    'SCOPE_SOURCE',
    'SCOPE_VALUES',
    'CSQ_PARENTHESES_ALWAYS',
    'CSQ_PARENTHESES_UNNESTED',
    'CSQ_PARENTHESES_NEVER',
    'SNAKE_CASE_STEP1',
    'SNAKE_CASE_STEP2',
    'MODEL_BASE'
]

SENTINEL = object()

#: Operations for use in SQL expressions.
OP = attrdict(
    AND='AND',
    OR='OR',
    ADD='+',
    SUB='-',
    MUL='*',
    DIV='/',
    BIN_AND='&',
    BIN_OR='|',
    XOR='#',
    MOD='%',
    EQ='=',
    LT='<',
    LTE='<=',
    GT='>',
    GTE='>=',
    NE='!=',
    IN='IN',
    NOT_IN='NOT IN',
    IS='IS',
    IS_NOT='IS NOT',
    LIKE='LIKE',
    ILIKE='ILIKE',
    BETWEEN='BETWEEN',
    REGEXP='REGEXP',
    IREGEXP='IREGEXP',
    CONCAT='||',
    BITWISE_NEGATION='~')


def expression_func(op):
    def f(l, r):
        from .pw import Expression

        return Expression(l, op, r)

    return f


# To support "django-style" double-underscore filters, create a mapping between
# operation name and operation code, e.g. "__eq" == OP.EQ.
DJANGO_MAP = attrdict({
    'eq': operator.eq,
    'lt': operator.lt,
    'lte': operator.le,
    'gt': operator.gt,
    'gte': operator.ge,
    'ne': operator.ne,
    'in': operator.lshift,
    'is': expression_func(OP.IS),
    'like': expression_func(OP.LIKE),
    'ilike': expression_func(OP.ILIKE),
    'regexp': expression_func(OP.REGEXP),
})

#: Mapping of field type to the data-type supported by the database. Databases
#: may override or add to this list.
FIELD = attrdict(
    AUTO='INTEGER',
    BIGAUTO='BIGINT',
    BIGINT='BIGINT',
    BLOB='BLOB',
    BOOL='SMALLINT',
    CHAR='CHAR',
    DATE='DATE',
    DATETIME='DATETIME',
    DECIMAL='DECIMAL',
    DEFAULT='',
    DOUBLE='REAL',
    FLOAT='REAL',
    INT='INTEGER',
    SMALLINT='SMALLINT',
    TEXT='TEXT',
    TIME='TIME',
    UUID='TEXT',
    UUIDB='BLOB',
    VARCHAR='VARCHAR')

#: Join helpers (for convenience) -- all join types are supported, this object
#: is just to help avoid introducing errors by using strings everywhere.
JOIN = attrdict(
    INNER='INNER JOIN',
    LEFT_OUTER='LEFT OUTER JOIN',
    RIGHT_OUTER='RIGHT OUTER JOIN',
    FULL='FULL JOIN',
    FULL_OUTER='FULL OUTER JOIN',
    CROSS='CROSS JOIN',
    NATURAL='NATURAL JOIN',
    LATERAL='LATERAL',
    LEFT_LATERAL='LEFT JOIN LATERAL')

# Row representations.
ROW = attrdict(
    TUPLE=1,
    DICT=2,
    NAMED_TUPLE=3,
    CONSTRUCTOR=4,
    MODEL=5
)

SCOPE_NORMAL = 1
SCOPE_SOURCE = 2
SCOPE_VALUES = 4
SCOPE_CTE = 8
SCOPE_COLUMN = 16

# Rules for parentheses around subqueries in compound select.
CSQ_PARENTHESES_NEVER = 0
CSQ_PARENTHESES_ALWAYS = 1
CSQ_PARENTHESES_UNNESTED = 2

# Regular expressions used to convert class names to snake-case table names.
# First regex handles acronym followed by word or initial lower-word followed
# by a capitalized word. e.g. APIResponse -> API_Response / fooBar -> foo_Bar.
# Second regex handles the normal case of two title-cased words.
SNAKE_CASE_STEP1 = re.compile('(.)_*([A-Z][a-z]+)')
SNAKE_CASE_STEP2 = re.compile('([a-z0-9])_*([A-Z])')

# Helper functions that are used in various parts of the codebase.
MODEL_BASE = '_metaclass_helper_'

# -*- coding: utf-8 -*-
# @Time: 2023/6/8 17:11
from .pw import SQL, Context, CommaNodeList, NodeList, Value, Field, Node, fn
from .utils import basestring
from .utils import ensure_entity


def mysql_conflict_statement(on_conflict, query):
    if not on_conflict._action: return

    action = on_conflict._action.lower()
    if action == 'replace':
        return SQL('REPLACE')
    elif action == 'ignore':
        return SQL('INSERT IGNORE')
    elif action != 'update':
        raise ValueError('Un-supported action for conflict resolution. '
                         'MySQL supports REPLACE, IGNORE and UPDATE.')


def mysql_conflict_update(on_conflict, query):
    if on_conflict._where or on_conflict._conflict_target or \
            on_conflict._conflict_constraint:
        raise ValueError('MySQL does not support the specification of '
                         'where clauses or conflict targets for conflict '
                         'resolution.')

    updates = []
    if on_conflict._preserve:
        # Here we need to determine which function to use, which varies
        # depending on the MySQL server version. MySQL and MariaDB prior to
        # 10.3.3 use "VALUES", while MariaDB 10.3.3+ use "VALUE".
        version = self.server_version or (0,)
        if version[0] == 10 and version >= (10, 3, 3):
            VALUE_FN = fn.VALUE
        else:
            VALUE_FN = fn.VALUES

        for column in on_conflict._preserve:
            entity = ensure_entity(column)
            expression = NodeList((
                ensure_entity(column),
                SQL('='),
                VALUE_FN(entity)))
            updates.append(expression)

    if on_conflict._update:
        for k, v in on_conflict._update.items():
            if not isinstance(v, Node):
                # Attempt to resolve string field-names to their respective
                # field object, to apply data-type conversions.
                if isinstance(k, basestring):
                    k = getattr(query.table, k)
                if isinstance(k, Field):
                    v = k.to_value(v)
                else:
                    v = Value(v, unpack=False)
            updates.append(NodeList((ensure_entity(k), SQL('='), v)))

    if updates:
        return NodeList((SQL('ON DUPLICATE KEY UPDATE'),
                         CommaNodeList(updates)))


def mysql_default_values_insert(ctx):
    return ctx.literal('() VALUES ()')


MYSQLContext = Context(**{
    'field_types': {'AUTO': 'INTEGER AUTO_INCREMENT', 'BIGAUTO': 'BIGINT AUTO_INCREMENT', 'BIGINT': 'BIGINT',
                    'BLOB': 'BLOB', 'BOOL': 'BOOL', 'CHAR': 'CHAR', 'DATE': 'DATE', 'DATETIME': 'DATETIME',
                    'DECIMAL': 'NUMERIC', 'DEFAULT': '', 'DOUBLE': 'DOUBLE PRECISION', 'FLOAT': 'FLOAT',
                    'INT': 'INTEGER', 'SMALLINT': 'SMALLINT', 'TEXT': 'TEXT', 'TIME': 'TIME', 'UUID': 'VARCHAR(40)',
                    'UUIDB': 'VARBINARY(16)', 'VARCHAR': 'VARCHAR'},
    'operations': {'AND': 'AND', 'OR': 'OR', 'ADD': '+', 'SUB': '-', 'MUL': '*', 'DIV': '/', 'BIN_AND': '&',
                   'BIN_OR': '|', 'XOR': 'XOR', 'MOD': '%', 'EQ': '=', 'LT': '<', 'LTE': '<=', 'GT': '>', 'GTE': '>=',
                   'NE': '!=', 'IN': 'IN', 'NOT_IN': 'NOT IN', 'IS': 'IS', 'IS_NOT': 'IS NOT', 'LIKE': 'LIKE BINARY',
                   'ILIKE': 'LIKE', 'BETWEEN': 'BETWEEN', 'REGEXP': 'REGEXP BINARY', 'IREGEXP': 'REGEXP',
                   'CONCAT': '||', 'BITWISE_NEGATION': '~'}, 'param': '%s', 'quote': '``',
    'compound_select_parentheses': 2, 'for_update': True, 'index_schema_prefix': False,
    'index_using_precedes_table': True, 'limit_max': 18446744073709551615, 'nulls_ordering': False,
    'conflict_statement': mysql_conflict_statement, 'conflict_update': mysql_conflict_update,
    'default_values_insert': mysql_default_values_insert
})

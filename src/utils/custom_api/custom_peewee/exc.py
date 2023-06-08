# -*- coding: utf-8 -*-
# @Time: 2023/6/8 17:12
class PeeweeException(Exception):
    def __init__(self, *args):
        if args and isinstance(args[0], Exception):
            self.orig, args = args[0], args[1:]
        super(PeeweeException, self).__init__(*args)


class DoesNotExist(PeeweeException):
    pass


class DatabaseError(PeeweeException):
    pass


class MultiDeleteNotAllowed(PeeweeException):
    pass


class MultiUpdateNotAllowed(PeeweeException):
    pass


class IntegrityError(DatabaseError):
    pass


__all__ = [
    'PeeweeException',
    'DatabaseError',
    'MultiDeleteNotAllowed',
    'MultiUpdateNotAllowed',
    'IntegrityError',
    'DoesNotExist'
]

# -*- coding: utf-8 -*-
# @Time: 2023/6/8 17:13
class attrdict(dict):

    def __getattr__(self, attr):
        try:
            return self[attr]
        except KeyError:
            raise AttributeError(attr)

    def __setattr__(self, attr, value):
        self[attr] = value

    def __iadd__(self, rhs):
        self.update(rhs)
        return self

    def __add__(self, rhs):
        d = attrdict(self)
        d.update(rhs)
        return d

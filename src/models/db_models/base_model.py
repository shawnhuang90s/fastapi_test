# -*- coding: utf-8 -*-
# @Time: 2023/6/8 17:15
import time
from utils.custom_api.custom_peewee import *


class BaseModel(Model):
    create_time = BigIntegerField(null=False, default=lambda: int(time.time() * 1000))
    update_time = BigIntegerField(null=False, default=lambda: int(time.time() * 1000))

    @classmethod
    def update(cls, __data=None, **update):
        if not update.get('update_time', 0):
            update['update_time'] = int(time.time() * 1000)
        return super(BaseModel, cls).update(__data, **update)

    def save(self, force_insert=False, only=None):
        self.update_time = int(time.time() * 1000)
        return super(BaseModel, self).save(force_insert, only)

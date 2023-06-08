# -*- coding: utf-8 -*-
# @Time: 2023/6/8 17:02
import time
from dao.db import TestMysqlConn
from utils.custom_api.custom_peewee import *
from models.db_models.base_model import BaseModel


class HTTPMethodField(SmallIntegerField):

    def db_value(self, value):
        str2int = {
            'ALL': 0,
            'GET': 1,
            'POST': 2,
            'HEAD': 3,
            'OPTIONS': 4,
            'PUT': 5,
            'PATCH': 6,
            'DELETE': 7,
            'TRACE': 8,
            'CONNECT': 9,
        }
        return str2int[value.upper()]

    def python_value(self, value):
        int2str = {
            0: 'ALL',
            1: 'GET',
            2: 'POST',
            3: 'HEAD',
            4: 'OPTIONS',
            5: 'PUT',
            6: 'PATCH',
            7: 'DELETE',
            8: 'TRACE',
            9: 'CONNECT',
        }
        return int2str[value]


class UserInfo(BaseModel):
    id = BigAutoField(null=False)
    username = CharField(max_length=20, unique=True, null=False)
    password = CharField(max_length=16, null=False)
    email = CharField(max_length=32)
    is_active = SmallIntegerField(null=False, default=1)
    is_superuser = SmallIntegerField(null=False, default=0)
    last_reset_time = BigIntegerField(null=False, default=lambda: int(time.time() * 1000))
    last_login_time = BigIntegerField(null=False, default=lambda: int(time.time() * 1000))

    class Meta:
        table_name = "test_user_info"
        connection_factory = TestMysqlConn
        indexes = (
            ("username", True),
            # CREATE INDEX idx_username_password ON test_user_info (username, password);
            (("username", "password", False))
        )


class Permission(BaseModel):
    id = BigAutoField()
    name = CharField(150)
    backend_url = TextField(default='')
    method = HTTPMethodField(null=False, default='ALL')
    parent = BigIntegerField(null=False, default=-1)

    class Meta:
        table_name = 'test_permission'
        connection_factory = TestMysqlConn


class UserGroup(BaseModel):
    id = BigAutoField()
    user_id = BigIntegerField(null=False)
    group_id = BigIntegerField(null=False)

    class Meta:
        table_name = 'test_user_group'
        connection_factory = TestMysqlConn


class UserPermission(BaseModel):
    id = BigAutoField()
    user_id = BigIntegerField(null=False)
    permission_id = BigIntegerField(null=False)

    class Meta:
        table_name = 'test_user_permission'
        connection_factory = TestMysqlConn


class GroupPermission(BaseModel):
    id = BigAutoField()
    group_id = BigIntegerField()
    permission_id = BigIntegerField()

    class Meta:
        table_name = 'test_group_permission'
        connection_factory = TestMysqlConn


class Group(BaseModel):
    id = BigAutoField()
    name = CharField(max_length=150)
    creator = BigIntegerField(default=-1)

    class Meta:
        table_name = 'test_group'
        connection_factory = TestMysqlConn


if __name__ == '__main__':
    import asyncio

    async def query_user_info(username):
        user_obj = await UserInfo.select().where(
            UserInfo.username == username
        ).single().execute()
        print(user_obj)

        id_list = await UserInfo.select(UserInfo.id).many_value().execute()
        print(id_list)

    loop = asyncio.get_event_loop()
    loop.run_until_complete(query_user_info('shawn'))

# -*- coding: utf-8 -*-
# @Time: 2023/6/8 17:03
import asyncio
from config.mysql_config import MySQLConn, PoolWrapper, MySQLConfig


class TestMysqlConn(MySQLConn):

    pool = None

    def __init__(self):
        if TestMysqlConn.pool is None:
            TestMysqlConn.pool = PoolWrapper(mincached=MySQLConfig.MIN_CACHED,
                                             maxcached=MySQLConfig.MAX_CACHED,
                                             minsize=MySQLConfig.MINSIZE,
                                             maxsize=MySQLConfig.MAXSIZE,
                                             loop=asyncio.get_running_loop(),
                                             echo=False,
                                             pool_recycle=MySQLConfig.POOL_RECYCLE,
                                             host=MySQLConfig.HOST,
                                             user=MySQLConfig.USER,
                                             password=MySQLConfig.PASSWORD,
                                             db=MySQLConfig.DB,
                                             port=MySQLConfig.PORT)
        super(TestMysqlConn, self).__init__()

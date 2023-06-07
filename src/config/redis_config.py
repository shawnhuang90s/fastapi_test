# -*- coding: utf-8 -*-
# @Time: 2023/6/7 10:22
import aioredis
import logging


class InitRedisConfig:

    def __init__(self):
        self.RedisConn = aioredis.from_url("redis://localhost", db=0, decode_responses=True)
        logging.info(f"{self.RedisConn} 连接成功.")


REDIS_CONN = InitRedisConfig().RedisConn

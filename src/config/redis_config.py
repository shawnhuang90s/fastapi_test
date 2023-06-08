# -*- coding: utf-8 -*-
# @Time: 2023/6/7 10:22
import aioredis
import logging


class InitRedisConfig:

    def __init__(self):
        self.RedisConn = aioredis.from_url("redis://localhost", db=0, decode_responses=True)
        logging.info(f"{self.RedisConn} successfully connected.")


REDIS_CONN = InitRedisConfig().RedisConn


class RedisKeys:
    LOCKED_USER_KEY = "locked_user: "
    FAILURE_LOGIN_KEY = "failure_login_user: "
    USER_ID_SESSION_KEY = "user_id_session: "
    USER_SESSION_ID_KEY = "user_session_id: "
    USER_ID_PERMISSION_KEY = "user_id_permission: "


class RedisConfig:
    LOCK_TIMES = 5  # If there are 10 consecutive login errors, the account will be locked for 30 minutes.
    LOCK_TIME = 1 * 60  # Lock time
    WARNING_TIMES = 9  # If there are 19 consecutive login errors, the account will be warned and frozen.
    FROZEN_TIMES = 10  # If there are 20 consecutive login errors, the account will be frozen.
    USER_SESSION_EXPIRE = 16 * 60 * 60  # Keep user session information for 16 hours, and log in again if it expires.

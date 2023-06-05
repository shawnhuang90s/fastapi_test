# -*- coding: utf-8 -*-
# @Time: 2023/6/5 10:43
import os
from enum import Enum, IntEnum


class ServerInfo:
    ENV = os.environ.get("ENV")
    HOST = "0.0.0.0"
    PORT = 8888


class Env:
    DEV = "dev"
    TEST = "test"
    PRE = "pre"
    PROD = "prod"

    IS_PROD_ENV = [PRE, PROD]

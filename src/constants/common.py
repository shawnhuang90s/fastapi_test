# -*- coding: utf-8 -*-
# @Time: 2023/6/5 10:43
import os


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


class ReturnTips:
    INVALID_USERNAME = 'The username contains illegal characters or is not between 4 and 15 in length.'
    USERNAME_EXIST = 'User name already exists.'
    SUFFIX_HINT = 'The file suffix must be csv or CSV.'
    PWD_LENGTH_ERROR = 'The password length set is too short.'
    PWD_NUMBERS_LENGTH_ERROR = 'The password set contains too few digits'
    PWD_LETTERS_LENGTH_ERROR = 'The password set contains too few letters'
    PWD_SPECIALS_ERROR = 'The password set must contain at least one special character'


class UserStatus:
    IS_ACTIVE = 1
    IS_INACTIVE = 0

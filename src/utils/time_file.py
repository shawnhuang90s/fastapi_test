# -*- coding: utf-8 -*-
# @Time : 2023/6/5 16:45
import time


def get_str_by_timestamp(timestamp_obj, format_str="%Y-%m-%d %H:%M:%S"):
    """Convert timestamp to string format"""
    str_time = time.strftime(format_str, time.localtime(timestamp_obj / 1000))
    return str_time


def current_timestamp():
    return int(time.time() * 1000)

# -*- coding: utf-8 -*-
# @Time: 2023/6/5 15:55
import random
import socket
import os
import time
import logging

SEQUENCE_BITS = 10
SEQUENCE_MASK = -1 ^ (-1 << SEQUENCE_BITS)
WORKER_ID_BITS = 12
MAX_WORKER_ID = -1 ^ (-1 << WORKER_ID_BITS)
TWITTER_FIRST_TIMESTAMP = 1288834974657
TIMESTAMP_LEFT_SHIFT = SEQUENCE_BITS + WORKER_ID_BITS
WORKER_ID_SHIFT = SEQUENCE_BITS


def get_local_ip():
    hostname = socket.gethostname()
    ip = socket.gethostbyname(hostname)
    return ip


def get_current_pid():
    return os.getpid()


class InvalidSystemClock(Exception):
    pass


class Singleton(type):

    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class SnowFlakeID(metaclass=Singleton):

    INIT_SEQ = random.randint(1, SEQUENCE_MASK)

    def __init__(self, ip: str = get_local_ip(), pid_or_port: int = get_current_pid()):
        worker_id_gen = hash(f"{ip}:{pid_or_port}")
        self.worker_id = worker_id_gen % MAX_WORKER_ID
        self.sequence = SnowFlakeID.INIT_SEQ
        self.last_timestamp = -1

    @staticmethod
    def _gen_timestamp():
        return int(time.time() * 1000)

    def _til_next_millis(self, last_timestamp):
        timestamp = self._gen_timestamp()
        while timestamp <= last_timestamp:
            timestamp = self._gen_timestamp()
        return timestamp

    def get_id(self):
        timestamp = self._gen_timestamp()
        if timestamp < self.last_timestamp:
            logging.error(f"clock is moving backwards. Rejecting requests until {self.last_timestamp}")
            raise InvalidSystemClock
        elif timestamp == self.last_timestamp:
            self.sequence = (self.sequence + 1) & SEQUENCE_MASK
            if self.sequence == self.INIT_SEQ:
                timestamp = self._til_next_millis(self.last_timestamp)
        else:
            self.sequence = SnowFlakeID.INIT_SEQ + random.randint(0, 1)
        self.last_timestamp = timestamp
        new_id = ((timestamp - TWITTER_FIRST_TIMESTAMP) << TIMESTAMP_LEFT_SHIFT) | (
                self.worker_id << WORKER_ID_SHIFT) | self.sequence
        return new_id


if __name__ == "__main__":
    from utils.custom_print import print
    print(SnowFlakeID().get_id())

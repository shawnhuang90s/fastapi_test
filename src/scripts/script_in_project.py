# -*- coding: utf-8 -*-
# @Time: 2023/6/9 11:48
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

import asyncio
import logging

from config.http_config import Http
from config.system_config import ServiceConf
from config.log_config import init_logging_config


async def test_process():
    try:
        host = ServiceConf.test_host
        url = f"{host}/"
        async with Http() as http:
            res = await http.aio_get(url)
            logging.info(f"Test url: {url}, resp: {res.text}, status: {res.status}")
    except Exception:
        logging.error(f'Failed to execute test func', exc_info=True)


if __name__ == '__main__':
    # Examples of cmd execution commands for Windows:
    # C:\Users\huangxy4\Envs\fastapi_test\Scripts\python.exe E:\learning\fastapi_test\src\scripts\script_in_project.py

    # Example of executing commands in Linux:
    # /data/venv/fastapi_test/bin/python3  /data/app/fastapi_test/src/scripts/script_in_project.py --env=prod
    init_logging_config()
    loop = asyncio.get_event_loop()
    loop.run_until_complete(test_process())

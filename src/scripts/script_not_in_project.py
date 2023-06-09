# -*- coding: utf-8 -*-
# @Time: 2023/6/9 15:36
import os
import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

import asyncio
from urllib import parse
import logging
from typing import Dict
import aiohttp
from pydantic import BaseModel, Field
import logging.config
from pathlib import Path

logger = logging.getLogger()
PROJECT_NAME = "fastapi_test"
ROOT_LOG_PATH = "/data/logs"


class ServiceConf:
    test_host = 'http://127.0.0.1:8888'


class HttpResp(BaseModel):
    status: int = Field(description="Http status")
    text: str = Field(description="Http content")


class Http:

    session = None

    def __init__(self):
        if Http.session is None:
            async def on_request_start(session, trace_config_ctx, params) -> None:
                trace_config_ctx.start = asyncio.get_running_loop().time()

            async def on_request_end(session, trace_config_ctx, params) -> None:
                elapsed = asyncio.get_running_loop().time() - trace_config_ctx.start
                real_url = str(params.response.real_url)
                msg = {}
                msg["enter_req_name"] = "Fastapi_test_request_index"
                msg["enter_req_url"] = str(params.url)
                msg["enter_req_status"] = params.response.status
                msg["enter_req_elapsed"] = elapsed
                msg["enter_req_params"] = {}
                msg["enter_req_headers"] = params.headers()
                response_content = await params.response.read()
                msg["enter_req_response"] = response_content.decode("utf-8")
                if real_url.find("?") > 0:
                    msg["enter_req_params"] = dict(parse.parse_qsl(parse.urlsplit(real_url).query))
                logging.info(f"enter_req_info: {msg}", extra=msg)

            trace_config = aiohttp.TraceConfig()
            trace_config.on_request_start.append(on_request_start)
            trace_config.on_request_end.append(on_request_end)
            Http.session = aiohttp.ClientSession(
                loop=asyncio.get_running_loop(),
                trace_configs=[trace_config]
            )

        super(Http, self).__init__()

    @classmethod
    async def close(cls):
        if cls.session is not None:
            await cls.session.close()

    async def aio_get(
            self,
            url: str,
            params: Dict[str, any] = None,
            headers: Dict[str, str] = None,
            timeout: float = 60
    ) -> HttpResp:
        try:
            async with self.session.get(url, params=params, headers=headers, timeout=timeout) as response:
                status = response.status
                text = await response.text()
                return HttpResp(status=status, text=text)
        except Exception as e:
            logging.error(f"aio_get error: {e}", exc_info=True)
            raise e

    async def aio_post(
            self,
            url: str,
            data: Dict[str, any],
            headers: Dict[str, str] = None,
            timeout: float = 60
    ) -> HttpResp:
        try:
            async with self.session.post(url, data=data, timeout=timeout, headers=headers) as response:
                status = response.status
                text = await response.text()
                return HttpResp(status=status, text=text)
        except Exception as e:
            logging.error(f"aio_post error: {e}", exc_info=True)
            raise e

    async def aio_put(
            self,
            url: str,
            data: Dict[str, any],
            headers: Dict[str, str] = None,
            timeout: float = 60
    ) -> HttpResp:
        try:
            async with self.session.put(url, data=data, timeout=timeout, headers=headers) as response:
                status = response.status
                text = await response.text()
                return HttpResp(status=status, text=text)
        except Exception as e:
            logging.error(f"aio_put error: {e}", exc_info=True)
            raise e

    async def aio_post_json(
            self,
            url: str,
            data: Dict[str, any],
            headers: Dict[str, str] = None,
            timeout: float = 60
    ) -> HttpResp:
        try:
            async with self.session.post(url, json=data, timeout=timeout, headers=headers) as response:
                status = response.status
                text = await response.text()
                return HttpResp(status=status, text=text)
        except Exception as e:
            logging.error(f"aio_post_json error: {e}", exc_info=True)
            raise e

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass


def init_logging_config(path=''):
    root_path = Path(ROOT_LOG_PATH).joinpath(PROJECT_NAME)
    if path:
        root_path = root_path.joinpath(path)

    if not Path(root_path).exists():
        Path(root_path).mkdir(parents=True)

    root_log_file = root_path.joinpath("root.log")
    root_json_log_file = root_path.joinpath("root_json.log")

    logging_settings = {
        'version': 1,
        'formatters': {
            'simple': {
                'format': '{message}',
                'style': '{',
            },
            'normal': {
                'format': '{asctime} {levelname} {pathname} {lineno} {process:d} {thread:d} {message}',
                'style': '{',
            },
            "json": {
                "()": "pythonjsonlogger.jsonlogger.JsonFormatter",
                'format': '%(asctime) %(levelname) %(module) %(filename) %(pathname) '
                          '%(lineno) %(process) %(thread) %(message)',
            }
        },
        'handlers': {
            'console': {
                'level': 'INFO',
                'class': 'logging.StreamHandler',
                'formatter': 'normal'
            },
            'root': {
                'level': 'INFO',
                'class': 'logging.handlers.WatchedFileHandler',
                'filename': root_log_file,
                'formatter': 'normal'
            },
            'root_json': {
                'level': 'INFO',
                'class': 'logging.handlers.WatchedFileHandler',
                'filename': root_json_log_file,
                'formatter': 'json'
            }
        },
        # Self testing can add terminal log output display
        # And test servers and formal servers generally need to remove console configuration information
        'loggers': {
            '': {
                'handlers': ['console', 'root', 'root_json'],
                'level': 'INFO',
                'propagate': False,
            }
        },
    }
    logging.config.dictConfig(logging_settings)


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
    # If the script file is not in the project, two points to note:
    #     1.The virtual environment where Python executes the script must
    #       have a third-party package used for the script file.
    #     2.The script file cannot import a module from a certain file of the project,
    #       and only the code of the required module can be copied into the script file.
    #
    # Example of executing command:
    #     C:\Users\huangxy4\Envs\project-test\Scripts\python.exe E:\learning\project-test\src\scripts\script_not_in_project.py

    # The current file can be placed in any directory other than the current project
    # And then executed directly in Python using the correct virtual environment
    init_logging_config()
    loop = asyncio.get_event_loop()
    loop.run_until_complete(test_process())

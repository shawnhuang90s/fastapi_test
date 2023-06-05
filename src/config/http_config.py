# -*- coding: utf-8 -*-
# @Time: 2023/6/4 23:59
import asyncio
from urllib import parse
import logging
from typing import Dict
import aiohttp

from models.common_model import HttpResp


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
                msg["enter_req_name"] = "fastapi_test_request_index"
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

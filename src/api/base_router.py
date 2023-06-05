# -*- coding: utf-8 -*-
# @Time: 2023/6/5 15:53
import time
import logging
import contextvars
from typing import Any
from fastapi import HTTPException, APIRouter
from fastapi.responses import JSONResponse, Response
from fastapi.routing import APIRoute, Callable, Request, Coroutine
from starlette.status import HTTP_500_INTERNAL_SERVER_ERROR

from utils.snow_flake_id import SnowFlakeID

REQUEST_ID_CTX = contextvars.ContextVar('request_id')
SNOW_ID = SnowFlakeID().get_id()


class RespDurationRoute(APIRoute):

    def get_route_handler(self) -> Callable[[Request], Coroutine[Any, Any, Response]]:
        original_route_handler = super().get_route_handler()

        async def custom_route_handler(request: Request) -> Response:
            REQUEST_ID_CTX.set(SnowFlakeID().get_id())
            req_body = await request.body()
            logging.info(f"request_id: {REQUEST_ID_CTX.get()}")
            logging.info(f"request_url: {request.url}")
            logging.info(f'request_headers: {request.headers}')
            if not isinstance(req_body, bytes):
                logging.info(f"request_body: {req_body.decode('utf-8')}")

            before = int(time.time() * 1000)
            try:
                res: Response = await original_route_handler(request)
                return res
            except Exception as e:
                logging.error(f"get_route_handler error: {e}")
                raise HTTPException(status_code=HTTP_500_INTERNAL_SERVER_ERROR)
            finally:
                duration = int(time.time() * 1000) - before
                logging.info(f'request spend time: {duration}ms.')

        return custom_route_handler


class BaseAPIRouter(APIRouter):

    def __init__(self) -> None:
        super().__init__()
        self.route_class = RespDurationRoute
        self.default_response_class = JSONResponse

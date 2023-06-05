# -*- coding: utf-8 -*-
# @Time: 2023/6/4 23:52
import logging
import uvicorn
from fastapi import FastAPI
from typing import Callable
from starlette.middleware.cors import CORSMiddleware
from starlette.middleware.trustedhost import TrustedHostMiddleware

from config.http_config import Http
from config.log_config import init_logging_config
from config.doc_config import custom_open_api
from constants.common import ServerInfo, Env


def create_start_app_handler(app: FastAPI) -> Callable:
    async def start_app() -> None:
        pass
    return start_app


def create_stop_app_handle(app: FastAPI) -> Callable:
    async def stop_app() -> None:
        await Http.close()
        await app.state.redis.close()
    return stop_app


def init_middleware(app):
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"]
    )

    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["*"]
    )


def init_routers(app: FastAPI):
    from api.routers import router as test_router
    app.include_router(test_router)


def create_app() -> FastAPI:
    init_logging_config()
    if ServerInfo.ENV == Env.IS_PROD_ENV:
        app = FastAPI(title="fastapi_test", debug=False, version="1.0", docs_url=None, redoc_url=None)
    else:
        app = FastAPI(title="fastapi_test", debug=True, version="1.0")
    init_middleware(app)
    init_routers(app)
    custom_open_api(app, "fastapi test project api docs", "2023-06-05", "fastapi test project api docs", app.routes)
    app.add_event_handler("startup", create_start_app_handler(app))
    app.add_event_handler("shutdown", create_stop_app_handle(app))
    return app


if __name__ == "__main__":
    app = create_app()
    logging.info("************************ System Start Running *************************")
    logging.info(f"current env: {ServerInfo.ENV}, current port: {ServerInfo.PORT}")
    uvicorn.run(app, host=ServerInfo.HOST, port=ServerInfo.PORT, loop="auto")

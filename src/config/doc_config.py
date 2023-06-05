# -*- coding: utf-8 -*-
# @Time: 2023/6/5 10:48
from fastapi import FastAPI
from typing import Sequence, Dict
from starlette.routing import BaseRoute
from fastapi.openapi.utils import get_openapi


def custom_open_api(
        app: FastAPI,
        title: str,
        version: str,
        description: str,
        routes: Sequence[BaseRoute]
) -> Dict[str, any]:
    open_api_schema = get_openapi(
        title=title,
        version=version,
        description=description,
        routes=routes
    )
    app.open_api_schema = open_api_schema

    return open_api_schema

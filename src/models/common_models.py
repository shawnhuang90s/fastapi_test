# -*- coding: utf-8 -*-
# @Time: 2023/6/5 10:18
from pydantic import BaseModel, Field


class HttpResp(BaseModel):
    status: int = Field(description="Http status")
    text: str = Field(description="Http content")

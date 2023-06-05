# -*- coding: utf-8 -*-
# @Time: 2023/6/5 10:18
from pydantic import BaseModel, Field
from typing import Optional


class HttpResp(BaseModel):
    status: int = Field(description="Http status")
    text: str = Field(description="Http content")


class ErrorCode(BaseModel):
    error_code: Optional[str]
    error_msg: Optional[str]
    tip_msg: Optional[str]

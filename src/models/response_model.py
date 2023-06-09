# -*- coding: utf-8 -*-
# @Time : 2023/6/5 16:12
import time
from typing import Optional
from pydantic import BaseModel, Field


class ResponseBaseModel(BaseModel):
    success: bool = Field(description="Server status: true/false")
    sys_time: Optional[int] = Field(description="Server system time", default=int(time.time() * 1000))
    error_code: Optional[str] = Field(description="Error code", default="")
    error_msg: Optional[str] = Field(description="Error message", default="")
    tip_msg: Optional[str] = Field(description="Reminder info", default="")


class FailureResponseModel(ResponseBaseModel):
    success: bool = Field(default=False)


class SuccessResponseModel(ResponseBaseModel):
    success: bool = Field(default=True)


class TimeInfosDetail(BaseModel):
    current_time: str = Field(description="Current Indonesian time zone time")
    set_time: str = Field(description="The time point of the set Indonesian time zone")
    local_set_time: str = Field(description="Convert the set time point to the time zone of the server")


class TimeInfos(ResponseBaseModel):
    data: TimeInfosDetail = Field(default={}, description="Time zone time example")

# -*- coding: utf-8 -*-
# @Time : 2023/6/5 16:12
import time
from typing import Optional, List
from pydantic import BaseModel, Field


class ResponseBaseModel(BaseModel):
    success: bool = Field(description="服务状态 true-服务处理请求正常 false-服务处理请求失败")
    sys_time: Optional[int] = Field(description="服务器时间", default=int(time.time() * 1000))
    error_code: Optional[str] = Field(description="success 为 false 时错误状态码", default="")
    error_msg: Optional[str] = Field(description="success 为 false 时的错误信息", default="")
    tip_msg: Optional[str] = Field(description="给用户提示信息", default="")


class FailureResponseModel(ResponseBaseModel):
    success: bool = Field(default=False)


class SuccessResponseModel(ResponseBaseModel):
    success: bool = Field(default=True)


class TimeInfosDetail(BaseModel):
    current_time: str = Field(description="当前印尼时区时间")
    set_time: str = Field(description="设置的印尼时区的时间点")
    local_set_time: str = Field(description="将设置的时间点转成服务器所属时区时间")


class TimeInfos(ResponseBaseModel):
    data: TimeInfosDetail = Field(default={}, description="时区时间示例")


class RedisDetail(BaseModel):
    r_key: str = Field(description="Redis 的键")
    r_value: str = Field(description="Redis 的值")


class RedisInfo(ResponseBaseModel):
    data: RedisDetail = Field(default={}, description="Redis 使用示例")


class GetUsersInfoResponseSchema(SuccessResponseModel):
    class Data(BaseModel):
        id: int = Field(description="用户ID")
        username: str = Field(description="用户名")
        email: str = Field(description="邮件")
        is_active: int = Field(description="是否被冻结：0-是 1-否")
        is_superuser: int = Field(description="是否是超管")
        last_reset_time: int = Field(description="上次重置密码时间")
        last_login_time: int = Field(description="上次登陆时间")
        create_time: str = Field(description="创建时间")
        update_time: str = Field(description="更新时间")

    data: Optional[List[Data]] = Field(description="数据")


class GetUserGroupInfosResponseSchema(SuccessResponseModel):
    class Data(BaseModel):
        id: int = Field(description="用户ID")
        username: str = Field(description="用户名")
        group_names: str = Field(description="用户关联的组")
        email: str = Field(description="邮件")
        is_active: int = Field(description="是否被冻结：0-是 1-否")
        create_time: str = Field(description="创建时间")

    data: Optional[List[Data]] = Field(description="数据")

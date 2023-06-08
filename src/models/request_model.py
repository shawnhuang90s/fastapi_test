# -*- coding: utf-8 -*-
# @Time: 2023/6/8 16:49
from pydantic import BaseModel, Field
from typing import Optional


class CreateUserResponseSchema(BaseModel):
    group_ids: Optional[str] = Field(description='Group IDS')
    email: Optional[str] = Field(description='User email')
    username: str = Field(description='Username')
    password: str = Field(description='User Password')

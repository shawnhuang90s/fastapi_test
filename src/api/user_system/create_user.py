# -*- coding: utf-8 -*-
# @Time: 2023/6/8 16:48
import json
import logging
import re
import csv

from api.base_router import BaseAPIRouter
from constants.error_code import CodeConst, error_code_msg
from models.request_model import CreateUserResponseSchema
from models.response_model import FailureResponseModel, SuccessResponseModel
from constants.common import ReturnTips
from service.user_service import password_validation, check_username, insert_new_user, get_group_infos, \
    insert_user_group
from config.redis_config import REDIS_CONN
from fastapi import UploadFile, File
from typing import Optional

router = BaseAPIRouter()


@router.post("/fastapi_test/create_user", summary='Create new user', response_model=SuccessResponseModel)
async def create_user(info: CreateUserResponseSchema):
    """
    curl --location --request POST '127.0.0.1:8888/fastapi_test/create_user' \
    --header 'Content-Type: application/json' \
    --data-raw '{
        "username": "Michael",
        "password": "abcd1234.",
        "email": "mike@qq.com",
        "group_ids": "[3, 4, 6]"
    }'
    """
    try:
        if not re.fullmatch(r'[a-zA-Z0-9_.-]{4,50}', info.username):
            err_code_info = error_code_msg(CodeConst.A1003, tip_msg=ReturnTips.INVALID_USERNAME)
            return FailureResponseModel(**dict(err_code_info))
        exist_flag = await check_username(info.username)
        if exist_flag:
            err_code_info = error_code_msg(CodeConst.A1003, tip_msg=ReturnTips.USERNAME_EXIST)
            return FailureResponseModel(**dict(err_code_info))
        error_msg = password_validation(info.password, minimum_length=8, number_of_numbers=4, number_of_letters=3)
        if error_msg is not None:
            err_code_info = error_code_msg(CodeConst.A1003, tip_msg=error_msg)
            return FailureResponseModel(**dict(err_code_info))
        new_user_id = await insert_new_user(info.username, info.password, email=info.email)
        group_ids = json.loads(info.group_ids)
        if group_ids:
            group_info = await get_group_infos(group_ids)
            if group_info:
                for group_id in group_ids:
                    await insert_user_group(new_user_id, group_id)
        if info.email:
            REDIS_CONN.hmset('email_users', {info.username: info.password})
        return SuccessResponseModel()
    except Exception as e:
        logging.error(f'/fastapi_test/create_user error: {e}', exc_info=True)
        err_code_info = error_code_msg(CodeConst.B2001)
        return FailureResponseModel(**dict(err_code_info))


@router.post("/fastapi_test/batch_create_user", summary='Batch create new users',
             response_model=SuccessResponseModel)
async def get_users_info(
    file: UploadFile = File(..., description='File for batch creation of users.'),
    group_ids: Optional[str] = File(default='[]', description='Group IDs')
):
    """
    curl --location --request POST '127.0.0.1:8888/fastapi_test/batch_create_user' \
    --form 'file=@"/C:/Users/huangxy4/Desktop/create_test.csv"' \
    --form 'group_ids="[1, 2, 3]"'
    """
    try:
        file_name = file.filename
        group_ids = json.loads(group_ids)
        if not (file_name.endswith('.csv') or file_name.endswith('.CSV')):
            err_code_info = error_code_msg(CodeConst.A1003, tip_msg=ReturnTips.SUFFIX_HINT)
            return FailureResponseModel(**dict(err_code_info))
        contents = await file.read()
        file_data = csv.DictReader(contents.decode("gbk", "ignore").splitlines())
        failed_hints = """"""
        n = 0
        # If an error is reported:_ Csv. Error: line contains NUL
        # Save the current Excel file as a separate file, and then upload the saved Excel file
        for row in file_data:
            n += 1
            username = row['username']
            email = row['email']
            if not any([username, email]):
                failed_hints += f'Registration failed on line {n}: username and email cannot both be empty\n'
                continue
            if not username:
                username = email
            exist_flag = await check_username(username)
            if exist_flag:
                failed_hints += f'Registration failed on line {n}: {ReturnTips.USERNAME_EXIST}\n'
                continue
            init_password = username
            new_user_id = await insert_new_user(username, init_password, email=email)
            if group_ids:
                group_info = await get_group_infos(group_ids)
                if group_info:
                    for group_id in group_ids:
                        await insert_user_group(new_user_id, group_id)
            if email:
                REDIS_CONN.hmset('email_users', {username: init_password})
        if failed_hints:
            err_code_info = error_code_msg(CodeConst.A1003, tip_msg=str(failed_hints))
            return FailureResponseModel(**dict(err_code_info))
        else:
            return SuccessResponseModel()
    except Exception as e:
        logging.error(f'/fastapi_test/batch_create_user error: {e}', exc_info=True)
        err_code_info = error_code_msg(CodeConst.B2001)
        return FailureResponseModel(**dict(err_code_info))

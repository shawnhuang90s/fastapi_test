# -*- coding: utf-8 -*-
# @Time: 2023/6/8 16:58
import asyncio
import time
import math
from hashlib import md5
import string

from models.db_models.user_system_model import UserInfo, UserPermission, GroupPermission, UserGroup, Group
from config.redis_config import REDIS_CONN
from config.redis_config import RedisKeys
from constants.common import ReturnTips, UserStatus


async def get_user_info(username=None, pwd=None, pk=None):
    where_params = []
    if username is not None:
        where_params.append(UserInfo.username == username)
    if pwd is not None:
        salt_pwd = md5(pwd.encode('utf8')).hexdigest()
        where_params.append(UserInfo.password == salt_pwd)
    if pk is not None:
        where_params.append(UserInfo.id == pk)
    if where_params:
        user_info = await UserInfo.select().where(*where_params).single().execute()
    else:
        user_info = await UserInfo.select().execute()
    return user_info


async def freeze_user(pk):
    await UserInfo.update(is_active=0).where(UserInfo.id == pk).execute()


async def clear_login_cache(clear_keys):
    for key in clear_keys:
        REDIS_CONN.delete(key)


async def check_pwd_overdue(last_reset_time):
    interval_seconds = int(time.time()) * 1000 - last_reset_time
    interval_days = math.ceil(interval_seconds / (60 * 60 * 24 * 1000))  # Round up, i.e. 9.1 to 10
    return True if int(interval_days) > 90 else False


async def update_last_login_time(pk):
    await UserInfo.update(last_login_time=int(time.time()) * 1000).where(UserInfo.id == pk).execute()


async def get_permission_ids_by_user_id(user_id):
    permission_ids = await UserPermission.select(
        UserPermission.permission_id
    ).where(
        UserPermission.user_id == user_id
    ).many_value().execute()
    if not permission_ids:
        permission_ids = await GroupPermission.select(
            GroupPermission.permission_id
        ).where(
            GroupPermission.group_id.in_(UserGroup.select(UserGroup.group_id).where(UserGroup.user_id == user_id))
        ).many_value().execute()
    return list(set(permission_ids))


async def refresh_user_permission(user_id):
    permission_ids = await get_permission_ids_by_user_id(user_id)
    user_permission_key = f'{RedisKeys.USER_ID_PERMISSION_KEY}{user_id}'
    REDIS_CONN.delete(user_permission_key)
    if permission_ids:
        REDIS_CONN.sadd(user_permission_key, *permission_ids)
    # print(REDIS_CONN.smembers(user_permission_key))


async def update_pwd(pk, new_pwd):
    new_salt_pwd = md5(new_pwd.encode('utf8')).hexdigest()
    await UserInfo.update(
        password = new_salt_pwd,
        last_reset_time = int(time.time()) * 1000,
    ).where(
        UserInfo.id == pk
    ).execute()


def password_validation(password, minimum_length=0, number_of_numbers=0, number_of_letters=0, need_special=True):
    """
    Determine whether the password is legal.
    :param password: Original password
    :param minimum_length: specifies the length that cannot be less than
    :param number_of_numbers: specifies the number of digits that must be included
    :param number_of_letters: specifies the number of letters that must be included
    :param need_special: Must it contain special characters
    :return:
    """
    tip_msg = None
    if minimum_length > 0:
        if len(password) < minimum_length:
            tip_msg = f'{ReturnTips.PWD_LENGTH_ERROR}, cannot be less than {minimum_length} characters.'
            return tip_msg

    exist_numbers = 0
    exist_letters = 0
    exist_specials = 0
    for i in password:
        if i in string.digits:
            exist_numbers += 1
        if i in string.ascii_letters:
            exist_letters += 1
        if i in string.punctuation:
            exist_specials += 1
    if number_of_numbers > 0:
        if exist_numbers < number_of_numbers:
            tip_msg = f'{ReturnTips.PWD_NUMBERS_LENGTH_ERROR}, cannot be less than {number_of_numbers} characters.'
            return tip_msg
    if number_of_letters > 0:
        if exist_letters < number_of_letters:
            tip_msg = f'{ReturnTips.PWD_LETTERS_LENGTH_ERROR}, cannot be less than {number_of_letters} characters.'
            return tip_msg
    if need_special is True:
        if exist_specials == 0:
            tip_msg = f'{ReturnTips.PWD_SPECIALS_ERROR}'
            return tip_msg

    return tip_msg


async def check_username(username):
    exist_flag = await UserInfo.select().where(UserInfo.username == username).single().execute()
    return True if exist_flag else False


async def insert_new_user(username, password, email=None):
    salt_pwd = md5(password.encode('utf8')).hexdigest()
    insert_params = dict(
        username=username,
        password=salt_pwd,
    )
    if email:
        insert_params.update(email=email)
    new_user_id = await UserInfo.insert(**insert_params).execute()
    return new_user_id


async def get_group_infos(group_ids):
    group_info = await Group.select().where(Group.id.in_(group_ids)).many().execute()
    return group_info


async def insert_user_group(user_id, group_id):
    user_group_id = await UserGroup.insert(user_id=user_id, group_id=group_id).execute()
    return user_group_id


async def disable_user(user_id):
    await UserInfo.update(is_active=UserStatus.IS_INACTIVE).where(UserInfo.id == user_id).execute()


async def enable_user(user_id):
    await UserInfo.update(is_active=UserStatus.IS_ACTIVE).where(UserInfo.id == user_id).execute()


async def delete_user_permission(user_id):
    await UserInfo.delete().where(UserInfo.id == user_id).execute()


async def insert_user_permission(user_id, permission_ids):
    rows = [(user_id, permission_id) for permission_id in permission_ids]
    fields = [UserPermission.user_id, UserPermission.permission_id]
    await UserPermission.insert_many(rows=rows, fields=fields).execute()


async def get_group_users(group_id):
    group_users = await UserGroup.select(
        UserGroup.user_id
    ).where(
        UserGroup.group_id == group_id
    ).many_value().execute()
    return group_users


async def delete_group_users(group_id):
    exist_ids = await UserGroup.select(UserGroup.id).where(UserGroup.group_id == group_id).many_value().execute()
    for exist_id in exist_ids:
        await UserGroup.delete().where(UserGroup.id == exist_id).execute()


async def insert_group_users(group_id, exist_group_users):
    rows = [(user_id, group_id) for user_id in exist_group_users]
    fields = [UserGroup.user_id, UserGroup.group_id]
    await UserGroup.insert_many(rows=rows, fields=fields).execute()


async def refresh_users_permission(user_ids):
    tasks = [refresh_user_permission(user_id) for user_id in user_ids]
    await asyncio.gather(*tasks)

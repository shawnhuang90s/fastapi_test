# -*- coding: utf-8 -*-
# @Time: 2023/6/9 16:39
import pytz
import arrow
import logging
from datetime import datetime

from api.base_router import BaseAPIRouter
from constants.error_code import CodeConst, error_code_msg
from models.response_model import ResponseBaseModel, TimeInfos

router = BaseAPIRouter()


@router.get("/fastapi_test/time_zone", summary="Obtain the current time of a country or region",
            response_model=TimeInfos)
async def get_time_info():
    """
    curl --location --request GET '127.0.0.1:8888/fastapi_test/time_zone'
    :return:
    """
    try:
        ########## Obtain the current time of a certain time zone ##########
        date_format = "%Y-%m-%d %H:%M:%S"
        arrow_format = "YYYY-MM-DD HH:mm:ss"
        IDN_timezone = "Asia/Jakarta"  # Indonesian time zone, based on the capital Jakarta time zone
        current_time = datetime.now(pytz.timezone(IDN_timezone))
        ########## Set a certain time zone and time point ##########
        current_date = str(current_time.date())
        set_time = pytz.timezone(IDN_timezone).localize(datetime.strptime(f"{current_date} 07:00:00", date_format))
        ########## Convert the set time point to the timestamp of the server's time zone ##########
        local_set_time = arrow.get(set_time).to('local').format(arrow_format)
        # local_set_timestamp = int(datetime.strptime(local_set_time, date_format).timestamp()) * 1000
        data = {
            "current_time": current_time.strftime(date_format),
            "set_time": set_time.strftime(date_format),
            "local_set_time": local_set_time
        }
        return TimeInfos(success=True, data=data)
    except Exception as e:
        logging.error(f'get_time_info error: {e}', exc_info=True)
        err_code_info = error_code_msg(CodeConst.B2003)
        return ResponseBaseModel(
            success=False,
            error_code=err_code_info.error_code,
            error_msg=err_code_info.error_msg,
            tip_msg=err_code_info.tip_msg
        )

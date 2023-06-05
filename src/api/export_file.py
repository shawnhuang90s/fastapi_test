# -*- coding: utf-8 -*-
# @Time: 2023/6/5 15:53
import logging
from starlette.responses import JSONResponse, StreamingResponse
from api.base_router import BaseAPIRouter

from models.response_model import ResponseBaseModel
from utils.download_file import DownloadFile
from constants.error_code import CodeConst, error_code_msg

router = BaseAPIRouter()


@router.get("/")
async def hello_world():
    """Service health testing"""
    return "Hello world"


@router.get("/fastapi_test/export", summary="export file test", response_model=ResponseBaseModel)
async def export_data():
    """
    curl --location --request GET '127.0.0.1:8888/fastapi_test/export'
    :return:
    """
    try:
        # Set header column names and column widths
        header_list = [
            ['Num', 5],
            ['Name', 10],
            ['Gender', 10],
            ['Hobby', 10],
            ['Birthday', 20]
        ]
        data_list = [
            [1, 'Zhang san', 'Male', 'Basketball', '1994-12-15'],
            [2, 'Li si', 'Female', 'Football', '1994-04-03'],
            [3, 'Wang wu', 'Male', 'Ping Pong', '1994-09-13'],
            [4, 'Zhang san', 'Male', 'Basketball', '1994-12-15'],
            [5, 'Li si', 'Female', 'Football', '1994-04-03'],
            [6, 'Wang wu', 'Male', 'Ping Pong', '1994-09-13'],
            [7, 'Zhang san', 'Male', 'Basketball', '1994-12-15'],
            [8, 'Li si', 'Female', 'Football', '1994-04-03'],
            [9, 'Wang wu', 'Male', 'Ping Pong', '1994-09-13'],
            [10, 'Zhang san', 'Male', 'Basketball', '1994-12-15'],
            [11, 'Li si', 'Female', 'Football', '1994-04-03'],
            [12, 'Wang wu', 'Male', 'Ping Pong', '1994-09-13'],
            [13, 'Zhang san', 'Male', 'Basketball', '1994-12-15'],
            [14, 'Li si', 'Female', 'Football', '1994-04-03'],
            [15, 'Wang wu', 'Male', 'Ping Pong', '1994-09-13'],
        ]
        d_obj = DownloadFile(header_list, data_list)
        return d_obj.get_response(response_type=StreamingResponse)
        # return d_obj.get_response(response_type=FileResponse)
    except Exception as e:
        logging.error(f'export_data error: {e}', exc_info=True)
        err_code_info = error_code_msg(CodeConst.B2002)
        return ResponseBaseModel(
            success=False,
            error_code=err_code_info.error_code,
            error_msg=err_code_info.error_msg,
            tip_msg=err_code_info.tip_msg
        )

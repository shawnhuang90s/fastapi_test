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
    """服务健康检测"""
    return "hello world"


@router.get("/fastapi_test/export", summary="导出数据示例", response_model=ResponseBaseModel)
async def export_data():
    """
    curl --location --request GET '127.0.0.1:8888/fastapi_test/export'
    :return:
    """
    try:
        # 设置表头列名和列宽
        header_list = [
            ['序号', 5],
            ['姓名', 10],
            ['性别', 10],
            ['爱好', 10],
            ['生日', 20]
        ]
        data_list = [
            [1, '张三', '男', '篮球', '1994-12-15'],
            [2, '李四', '女', '足球', '1994-04-03'],
            [3, '王五', '男', '兵乓球', '1994-09-13'],
            [4, '张三', '男', '篮球', '1994-12-15'],
            [5, '李四', '女', '足球', '1994-04-03'],
            [6, '王五', '男', '兵乓球', '1994-09-13'],
            [7, '张三', '男', '篮球', '1994-12-15'],
            [8, '李四', '女', '足球', '1994-04-03'],
            [9, '王五', '男', '兵乓球', '1994-09-13'],
            [10, '张三', '男', '篮球', '1994-12-15'],
            [11, '李四', '女', '足球', '1994-04-03'],
            [12, '王五', '男', '兵乓球', '1994-09-13'],
            [13, '张三', '男', '篮球', '1994-12-15'],
            [14, '李四', '女', '足球', '1994-04-03'],
            [15, '王五', '男', '兵乓球', '1994-09-13'],
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

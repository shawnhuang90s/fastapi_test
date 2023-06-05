# -*- coding: utf-8 -*-
# @Time : 2023/6/5 16:20
from collections import namedtuple
from models.common_model import ErrorCode


class CodeConst:
    """
    错误码及提示信息
    提示信息中 error_msg 是通用的，tip_msg 是自定义的
    1 开头的错误码代表客户端错误
    2 开头的错误码代表服务器内部错误
    3 开头的错误码代表第三方服务错误
    """
    B2002 = ("2002", "导出数据失败")

    def __init__(self):
        self.check_duplicate_code()

    @classmethod
    def check_duplicate_code(cls):
        """
        校验新增错误码时是否与当前存在的重复
        新增时必须类似 ("1001", "Client Error")，并且每个元素都不能重复，包括 error code 和 error msg
        """
        exist_values = set()
        for k,v in cls.__dict__.items():
            check_l = [k.startswith("A"), k.startswith("B"), k.startswith("C")]
            if any(check_l):
                if not isinstance(v, tuple):
                    raise Exception("错误码或错误提示信息缺失.")
                if len(v) != 2:
                    raise Exception("只需要错误码和错误提示信息.")
                exist_value = [v[0] in exist_values, v[1] in exist_values]
                if any(exist_value):
                    raise Exception("错误码或错误提示信息已存在.")
                else:
                    exist_values.add(v[0])
                    exist_values.add(v[1])


def error_code_msg(error_code, tip_msg=""):
    t = namedtuple("ErrorCodeMsg", ["error_code", "error_msg", "tip_msg"])
    if not tip_msg:
        error_code_cls = t._make([error_code[0], error_code[1], error_code[1]])
    else:
        # error_code_cls = t._make([error_code[0], error_code[1], f"{error_code[1]}: {tip_msg}"])
        error_code_cls = t._make([error_code[0], error_code[1], tip_msg])

    return ErrorCode(
        error_code=error_code_cls.error_code,
        error_msg=error_code_cls.error_msg,
        tip_msg=error_code_cls.tip_msg
    )


if __name__ == "__main__":
    code_obj = CodeConst()
    error_info = error_code_msg(code_obj.B2002)
    print(error_info.__dict__)

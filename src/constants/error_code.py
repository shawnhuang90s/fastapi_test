# -*- coding: utf-8 -*-
# @Time : 2023/6/5 16:20
from collections import namedtuple
from models.common_model import ErrorCode


class CodeConst:
    """
    Error codes and prompt information
    Error in prompt error_msg is universal, tip_msg is customized
    The error code starting with 1 represents a client error
    The error code starting with 2 represents an internal error in the server
    The error code starting with 3 represents a third-party service error
    """
    A1001 = ("1001", "Client error.")
    A1002 = ("1002", "Login failed.")
    A1003 = ("1003", "Failed to register new user.")
    A1004 = ("1004", "Password reset failed.")
    A1005 = ("1005", "Failed to update user permissions.")

    B2001 = ("2001", "Service error.")
    B2002 = ("2002", "Export data failed.")
    B2003 = ("2003", "Failed to obtain time info.")
    B2004 = ("2004", "Failed to obtain Redis info.")

    C3001 = ("3001", "Third party service error.")

    def __init__(self):
        self.check_duplicate_code()

    @classmethod
    def check_duplicate_code(cls):
        """
        Verify if the newly added error code is a duplicate of the current one
        When adding, it must be similar to ("1001", "Client Error")
        And each element cannot be duplicated, including error code and error msg
        """
        exist_values = set()
        for k,v in cls.__dict__.items():
            check_l = [k.startswith("A"), k.startswith("B"), k.startswith("C")]
            if any(check_l):
                if not isinstance(v, tuple):
                    raise Exception("Missing error code or error message.")
                if len(v) != 2:
                    raise Exception("All you need is an error code and error message.")
                exist_value = [v[0] in exist_values, v[1] in exist_values]
                if any(exist_value):
                    raise Exception("Error code or error message already exists.")
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

# -*- coding: utf-8 -*-
# @Time: 2023/6/5 16:24
import sys
import time


def stdout_write(msg: str):
    sys.stdout.write(msg)
    sys.stdout.flush()


def custom_print(*args):
    line = sys._getframe().f_back.f_lineno
    file_name = sys._getframe(1).f_code.co_filename
    args = (str(arg) for arg in args)
    sys.stdout.write(f'{file_name}:{line}  {time.strftime("%H:%M:%S")}  \033[0;94m{"".join(args)}\033[0m\n')


print = custom_print  # Monkey patch


if __name__ == '__main__':
    custom_print(123, 'abc')
    print(456, 'def')

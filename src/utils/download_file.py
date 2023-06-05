# -*- coding: utf-8 -*-
# @Time : 2023/6/5 16:32
import os
import time
import xlwt
from io import BytesIO
from config import BASE_PATH
from typing import List, Tuple
from starlette.background import BackgroundTask
from utils.time_file import get_str_by_timestamp
from starlette.responses import FileResponse, StreamingResponse


class DownloadFile:

    def __init__(self, header_list: List[list], data_list: List, encoding: str="utf-8") -> None:
        """
        初始化，关于颜色的选择参考：from xlwt.Style import _colour_map_text
        :param header_list: 表头列名列表
            例1：默认列宽
            header_list = [
                ['序号'],  # 表格第0列[此列表头名称]
                ['姓名'],
                ['性别'],
                ['爱好'],
                ['生日']
            ]
            例2：自定义列宽（列宽值为 int 类型，比如 10 表示列宽为 10 个英文字符长度）
            header = [
                ['序号', 5],   # 表格第0列[此列表头名称,列宽]
                ['姓名', 10],  # 表格第1列[此列表头名称,列宽]
                ['性别', 10],
                ['爱好', 10],
                ['生日', 20]
            ]
        :param data_list: 要保存到表中的数据
        :param encoding: 字符编码
        """
        self.header_list = header_list
        self.data_list = data_list
        self.work_book = xlwt.Workbook(encoding=encoding)

    def set_alignment_colour(
        self, ali_horiz: str="align: horiz center", ali_vert: str="align: vert center",
        fore_colour: str="pattern: pattern solid, fore_colour pale_blue"
    ) -> Tuple:
        """
        设置表头表体的对齐方式和背景颜色
        :param ali_horiz: 水平居中
        :param ali_vert: 垂直居中
        :param fore_colour: 背景颜色，默认是蓝色
        :return:
        """
        style_header = xlwt.easyxf(f"{fore_colour};{ali_horiz};{ali_vert}")
        style_body = xlwt.easyxf(ali_vert)
        return style_header, style_body

    def set_row_height(self, sheet, row_num: int, height: int=20*20) -> None:
        """
        设置某行的行高
        :param row_num: 第几行，0 表示第一行（表头）
        :return:
        """
        sheet.row(row_num).height_mismatch = True  # 表示当前行单独设置
        sheet.row(row_num).height = height  # 具体行高

    def set_header(self, sheet, style_header) -> None:
        """
        设置表头及数据写入
        :param sheet: 表对象
        :return:
        """
        for col in self.header_list:
            sheet.write(0, self.header_list.index(col), str(col[0]), style_header)
            # 如果有自定义列宽
            if len(col) == 2:
                sheet.col(self.header_list.index(col)).width = 256 * col[1]  # 256 是基数，col[1] 代表英文字符数量
        self.set_row_height(sheet, 0)

    def set_body(self, sheet, style_body) -> None:
        """
        设置表体及数据写入
        :param sheet: 表对象
        :return:
        """
        row_index = 1
        for row_data in self.data_list:
            self.set_row_height(sheet, row_index)
            for content in self.header_list:
                col_index = self.header_list.index(content)
                # 第 row_index 行第 col_index 列写入的数据及样式
                sheet.write(row_index, col_index, row_data[col_index], style_body)
            row_index += 1

    def set_frozen(self, sheet, frozen_row: int=1, frozen_col: int=0):
        """
        设置冻结行列，通俗解释即比如使用鼠标滚轮查看数据时，被冻结的行列固定着，而没冻结的则随着滚轮上下移动
        :param frozen_row: 冻结行（默认首行）
        :param frozen_col: 冻结列（默认不冻结）
        :return:
        """
        sheet.set_panes_frozen('1')  # 设置冻结为真（是下面两个设置的前提条件）
        sheet.set_horz_split_pos(frozen_row)  # 冻结前 n 行
        sheet.set_vert_split_pos(frozen_col)  # 冻结前 n 列

    def data_input(self, sheet_name="sheet"):
        """Write data to the table"""
        new_sheet = self.work_book.add_sheet(sheet_name)
        style_header, style_body = self.set_alignment_colour()
        self.set_header(new_sheet, style_header)
        self.set_body(new_sheet, style_body)
        self.set_frozen(new_sheet)
        return self.work_book

    def save_file(self, file_name: str="default.xls"):
        """
        写入数据到新表中
        :param file_name: 新建的 excel 文件名
        :param sheet_name: 新建的表名
        :return:
        """
        work_book = self.data_input()
        file_dir = self.create_dirs()
        file_path = f"{file_dir}{file_name}"
        new_file_path = file_path.replace("/", "\\")
        work_book.save(new_file_path)

        return new_file_path

    @staticmethod
    def create_dirs():
        """确定项目中有保存 excel 文件的目录"""
        file_dir = f"{BASE_PATH}/src/files/"
        os.makedirs(file_dir, exist_ok=True)

        return file_dir

    def get_response(self, response_type):
        """
        选择不同的方式导出Excel文件，如果是数据流，使用StreamingResponse；如果是文件下载，选择 FileResponse
        """
        if response_type == FileResponse:
            # 文件名不仅是保存到 files/ 目录下的文件名，也是输出到前端的文件名
            today = get_str_by_timestamp(int(time.time()) * 1000, format_str="%Y-%m-%d")
            file_name = f"export_data_{today}.xls"
            new_file_path = self.save_file(file_name=file_name)
            return FileResponse(
                path=new_file_path,
                filename=file_name,
                # Excel 文件发送成功后，顺便直接在 files/ 目录下自动删除
                # 如果下面这行代码被注释掉，文件会保留在 files/ 目录下
                background=BackgroundTask(lambda: os.remove(new_file_path))
            )
        elif response_type == StreamingResponse:
            work_book = self.data_input()
            sio = BytesIO()
            work_book.save(sio)
            sio.seek(0)
            # 组装头部
            headers = {
                "content-type": "application/vnd.ms-excel",
                "content-disposition": f'attachment;filename=download.xls'
            }
            # 将浏览器作为流返回
            return StreamingResponse(sio, media_type='xls/xlsx', headers=headers)


if __name__ == "__main__":
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
    d = DownloadFile(header_list, data_list)
    d.save_file(file_name="测试文件.xls")

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
        Reference for color selection -> from xlwt.Style import _colour_map_text
        :param header_list: List of header column names
            Example 1: Default column width
            header_list = [
                ['Num'],  #  Column 0 of the table [header name of this column]
                ['Name'],
                ['Gender'],
                ['Hobby'],
                ['Birthday']
            ]
            Example 2: Custom column width, column width value is of type int
                       for example, 10 represents a column width of 10 English characters in length
            header = [
                ['Num', 5],   # Column 0 of the table [header name, column width]
                ['Name', 10],  # Column 1 of the table [header name, column width]
                ['Gender', 10],
                ['Hobby', 10],
                ['Birthday', 20]
            ]
        :param data_list: The data to be saved to the table
        :param encoding: Character encoding
        """
        self.header_list = header_list
        self.data_list = data_list
        self.work_book = xlwt.Workbook(encoding=encoding)

    def set_alignment_colour(
        self, ali_horiz: str="align: horiz center", ali_vert: str="align: vert center",
        fore_colour: str="pattern: pattern solid, fore_colour pale_blue"
    ) -> Tuple:
        """
        Set the alignment and background color of the header and body
        :param ali_horiz: Horizontally
        :param ali_vert: Vertical center
        :param fore_colour: Background color, default to blue
        :return:
        """
        style_header = xlwt.easyxf(f"{fore_colour};{ali_horiz};{ali_vert}")
        style_body = xlwt.easyxf(ali_vert)
        return style_header, style_body

    def set_row_height(self, sheet, row_num: int, height: int=20*20) -> None:
        """
        Set the height of a row
        :param row_num: What is the first row, where 0 represents the first row (header)
        :return:
        """
        sheet.row(row_num).height_mismatch = True  # Indicates that the current row is set separately
        sheet.row(row_num).height = height  # Specific row height

    def set_header(self, sheet, style_header) -> None:
        """
        Set header and data writing
        :param sheet:
        :return:
        """
        for col in self.header_list:
            sheet.write(0, self.header_list.index(col), str(col[0]), style_header)
            # If there is a custom column width
            if len(col) == 2:
                # 256 is the cardinal number, col[1] represents the number of English characters
                sheet.col(self.header_list.index(col)).width = 256 * col[1]
        self.set_row_height(sheet, 0)

    def set_body(self, sheet, style_body) -> None:
        """
        Setting Table Body and Data Writing
        :param sheet:
        :return:
        """
        row_index = 1
        for row_data in self.data_list:
            self.set_row_height(sheet, row_index)
            for content in self.header_list:
                col_index = self.header_list.index(content)
                sheet.write(row_index, col_index, row_data[col_index], style_body)
            row_index += 1

    def set_frozen(self, sheet, frozen_row: int=1, frozen_col: int=0):
        """
        Set frozen rows and columns, which is commonly explained as follows
        When using the mouse scroll wheel to view data, the frozen rows and columns are fixed
        While the unfrozen rows and columns move up and down with the scroll wheel
        :param frozen_row: Freeze rows (default first row)
        :param frozen_col: Freeze column (default not frozen)
        :return:
        """
        sheet.set_panes_frozen('1')  # Setting freeze to true (a prerequisite for the following two settings)
        sheet.set_horz_split_pos(frozen_row)  # Freeze the first n rows
        sheet.set_vert_split_pos(frozen_col)  # Freeze the first n columns

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
        Write data to a new table
        :param file_name: New excel file name
        :param sheet_name: New table name
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
        """Confirm that there is a directory for saving excel files in the project"""
        file_dir = f"{BASE_PATH}/src/files/"
        os.makedirs(file_dir, exist_ok=True)

        return file_dir

    def get_response(self, response_type):
        """
        Choose different ways to export Excel files
        If it is a data stream, use StreamingResponse
        If it is a file download, select FileResponse
        """
        if response_type == FileResponse:
            # The file name is not only the file name saved in the files/directory
            # But also the file name output to the front-end
            today = get_str_by_timestamp(int(time.time()) * 1000, format_str="%Y-%m-%d")
            file_name = f"export_data_{today}.xls"
            new_file_path = self.save_file(file_name=file_name)
            return FileResponse(
                path=new_file_path,
                filename=file_name,
                # After successfully sending the Excel file
                # It will be automatically deleted directly in the files/directory
                # If the following line of code is commented out, the file will remain in the files/directory
                background=BackgroundTask(lambda: os.remove(new_file_path))
            )
        elif response_type == StreamingResponse:
            work_book = self.data_input()
            sio = BytesIO()
            work_book.save(sio)
            sio.seek(0)
            headers = {
                "content-type": "application/vnd.ms-excel",
                "content-disposition": f'attachment;filename=download.xls'
            }
            # Return browser as stream 
            return StreamingResponse(sio, media_type='xls/xlsx', headers=headers)


if __name__ == "__main__":
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
    d = DownloadFile(header_list, data_list)
    d.save_file(file_name="test_export_file.xls")

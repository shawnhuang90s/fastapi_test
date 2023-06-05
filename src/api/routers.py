# -*- coding: utf-8 -*-
# @Time: 2023/6/5 18:20
from api.base_router import BaseAPIRouter
from api import export_file

router = BaseAPIRouter()
router.include_router(export_file.router, tags=["Export excel file"])

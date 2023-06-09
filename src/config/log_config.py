# -*- coding: utf-8 -*-
# @Time: 2023/6/5 10:40
import logging.config
from pathlib import Path

logger = logging.getLogger()
PROJECT_NAME = "fastapi_test"
ROOT_LOG_PATH = "/data/logs"


def init_logging_config(path=''):
    root_path = Path(ROOT_LOG_PATH).joinpath(PROJECT_NAME)
    if path:
        root_path = root_path.joinpath(path)

    if not Path(root_path).exists():
        Path(root_path).mkdir(parents=True)

    root_log_file = root_path.joinpath("root.log")
    root_json_log_file = root_path.joinpath("root_json.log")

    logging_settings = {
        'version': 1,
        'formatters': {
            'simple': {
                'format': '{message}',
                'style': '{',
            },
            'normal': {
                'format': '{asctime} {levelname} {pathname} {lineno} {process:d} {thread:d} {message}',
                'style': '{',
            },
            "json": {
                "()": "pythonjsonlogger.jsonlogger.JsonFormatter",
                'format': '%(asctime) %(levelname) %(module) %(filename) %(pathname) '
                          '%(lineno) %(process) %(thread) %(message)',
            }
        },
        'handlers': {
            'console': {
                'level': 'INFO',
                'class': 'logging.StreamHandler',
                'formatter': 'normal'
            },
            'root': {
                'level': 'INFO',
                'class': 'logging.handlers.WatchedFileHandler',
                'filename': root_log_file,
                'formatter': 'normal'
            },
            'root_json': {
                'level': 'INFO',
                'class': 'logging.handlers.WatchedFileHandler',
                'filename': root_json_log_file,
                'formatter': 'json'
            }
        },
        # Self testing can add terminal log output display
        # And test servers and formal servers generally need to remove console configuration information
        'loggers': {
            '': {
                'handlers': ['console', 'root', 'root_json'],
                'level': 'INFO',
                'propagate': False,
            }
        },
    }
    logging.config.dictConfig(logging_settings)

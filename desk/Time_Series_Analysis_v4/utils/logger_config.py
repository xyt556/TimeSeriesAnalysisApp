# utils/logger_config.py
"""
日志配置模块
"""

import logging
import sys
from config import Config


def setup_logger(name='RSAnalysis'):
    """设置日志器"""
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, Config.LOG_LEVEL))

    # 避免重复添加处理器
    if logger.handlers:
        return logger

    # 文件处理器
    file_handler = logging.FileHandler(
        Config.LOG_FILE,
        encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(file_formatter)

    # 控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(
        '%(levelname)s - %(message)s'
    )
    console_handler.setFormatter(console_formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


# 创建全局logger
logger = setup_logger()
# utils/time_utils.py
"""
时间处理工具模块
"""

import re
import datetime
import numpy as np
import pandas as pd
from utils.logger_config import logger


class TimeExtractor:
    """时间信息提取器"""

    @staticmethod
    def extract_time(filename):
        """从文件名中提取时间信息

        支持格式：
        - 年-儒略日: NDVI_2000_123.tif
        - 年-月: NDVI_2000_01.tif, NDVI_200001.tif
        - 年: NDVI_2000.tif
        - 月份名称: NDVI_Jan_2000.tif

        Args:
            filename: 文件名

        Returns:
            datetime对象或None
        """
        # 年-儒略日格式
        m = re.search(r'(19\d{2}|20\d{2})_(\d{3})', filename)
        if m:
            year = int(m.group(1))
            day_of_year = int(m.group(2))
            try:
                date = datetime.datetime(year, 1, 1) + datetime.timedelta(days=day_of_year - 1)
                return date
            except:
                return datetime.datetime(year, 1, 1)

        # 年-月格式
        m = re.search(r'(19\d{2}|20\d{2})_(\d{1,2})', filename)
        if m:
            year = int(m.group(1))
            month = int(m.group(2))
            if 1 <= month <= 12:
                return datetime.datetime(year, month, 1)

        # 年月连续格式
        m = re.search(r'(19\d{2}|20\d{2})(\d{2})', filename)
        if m:
            year = int(m.group(1))
            month = int(m.group(2))
            if 1 <= month <= 12:
                return datetime.datetime(year, month, 1)

        # 仅年份格式
        m = re.search(r'(19\d{2}|20\d{2})', filename)
        if m:
            year = int(m.group(0))
            return datetime.datetime(year, 1, 1)

        # 月份名称格式
        month_map = {
            'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
            'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
        }
        for month_name, month_num in month_map.items():
            if month_name in filename:
                m = re.search(r'(19\d{2}|20\d{2})', filename)
                if m:
                    year = int(m.group(0))
                    return datetime.datetime(year, month_num, 1)

        logger.warning(f"Cannot extract time from filename: {filename}")
        return None

    @staticmethod
    def convert_to_years(times):
        """将时间数组转换为年份数组

        Args:
            times: 时间数组

        Returns:
            年份数组
        """
        years = []
        for t in times:
            if isinstance(t, np.datetime64):
                try:
                    year = pd.to_datetime(str(t)).year
                    years.append(year)
                except:
                    years.append(2000)
            elif hasattr(t, 'year'):
                years.append(t.year)
            else:
                try:
                    years.append(int(t))
                except:
                    years.append(2000)
        return np.array(years)

    @staticmethod
    def detect_frequency(times):
        """检测时间序列频率

        Args:
            times: 时间数组

        Returns:
            频率描述字符串
        """
        if len(times) < 2:
            return "单期数据"

        try:
            dt1 = pd.to_datetime(str(times[0]))
            dt2 = pd.to_datetime(str(times[1]))
            days_diff = (dt2 - dt1).days

            if 28 <= days_diff <= 31:
                return "月度数据"
            elif 88 <= days_diff <= 93:
                return "季度数据"
            elif 360 <= days_diff <= 370:
                return "年度数据"
            elif 7 <= days_diff <= 8:
                return "周数据"
            elif days_diff == 1:
                return "日数据"
            else:
                return f"自定义频率 (~{days_diff}天)"
        except:
            return "未知频率"

    @staticmethod
    def format_time_range(times):
        """格式化时间范围

        Args:
            times: 时间数组

        Returns:
            格式化的时间范围字符串
        """
        try:
            start = pd.to_datetime(str(times[0])).strftime('%Y-%m-%d')
            end = pd.to_datetime(str(times[-1])).strftime('%Y-%m-%d')
            return f"{start} 至 {end}"
        except:
            return f"{times[0]} 至 {times[-1]}"

    @staticmethod
    def format_time_labels(times):
        """格式化时间标签数组

        Args:
            times: 时间数组

        Returns:
            格式化的时间标签列表
        """
        labels = []
        for t in times:
            if isinstance(t, np.datetime64):
                labels.append(np.datetime_as_string(t, unit='D'))
            else:
                labels.append(str(t))
        return labels
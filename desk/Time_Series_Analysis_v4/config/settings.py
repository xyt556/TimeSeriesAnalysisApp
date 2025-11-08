# config/settings.py
"""
配置文件
集中管理所有系统配置参数
"""

import matplotlib


class Config:
    """系统配置类"""

    # 版本信息
    VERSION = "3.0"
    APP_NAME = "时序遥感分析系统"
    AUTHOR = "@3S&ML"

    # 性能配置
    MAX_WORKERS = 4
    CHUNK_SIZE = {'x': 512, 'y': 512}

    # 数据配置
    DEFAULT_DPI = 150
    NODATA_VALUE = -9999.0

    # 分析参数默认值
    MK_SIGNIFICANCE = 0.05  # Mann-Kendall显著性水平
    BFAST_THRESHOLD = 2.0  # BFAST阈值
    STL_DEFAULT_PERIOD = 12  # STL默认周期

    # 预处理参数
    SMOOTH_WINDOW = 5
    SMOOTH_POLYORDER = 2
    OUTLIER_THRESHOLD = 3.0

    # 聚类参数
    CLUSTER_DEFAULT = 5

    # 动画参数
    ANIMATION_FPS = 2

    # 颜色方案
    COLORMAPS = {
        'diverging': 'RdBu_r',
        'sequential': 'viridis',
        'trend': 'RdYlGn',
        'cluster': 'tab10',
        'hot': 'hot',
        'cool': 'cool'
    }

    # UI配置
    WINDOW_SIZE = (1600, 950)
    LEFT_PANEL_WIDTH = 340

    # 日志配置
    LOG_FILE = 'rs_analysis.log'
    LOG_LEVEL = 'INFO'

    @staticmethod
    def setup_matplotlib():
        """配置matplotlib中文显示"""
        matplotlib.rcParams['font.sans-serif'] = [
            'SimHei', 'Microsoft YaHei', 'STHeiti', 'Arial Unicode MS'
        ]
        matplotlib.rcParams['axes.unicode_minus'] = False
        matplotlib.rcParams['figure.max_open_warning'] = 50


# 初始化matplotlib配置
Config.setup_matplotlib()
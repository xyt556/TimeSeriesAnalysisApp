# main_enhanced_part1.py - 增强版时序遥感分析系统（第一部分）
"""
时序遥感分析系统 V3.0 - 增强版
包含：数据预处理、动画生成、聚类分析、多进程加速等功能
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import ttkbootstrap as tb
from ttkbootstrap.constants import *
import tempfile
from pathlib import Path
import re
import xarray as xr
import rioxarray as rxr
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.animation as animation
import warnings
import datetime
import threading
import os
from PIL import Image, ImageTk
import io
import rasterio
from rasterio.io import MemoryFile
from rasterio.transform import from_origin
import pandas as pd
from scipy import stats, fftpack
from scipy.signal import savgol_filter
from scipy.interpolate import CubicSpline
from statsmodels.tsa.seasonal import STL
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import zipfile
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import logging
import json
import pickle

warnings.filterwarnings('ignore')

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('rs_analysis.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'STHeiti', 'Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['figure.max_open_warning'] = 50


# ==================== 配置常量 ====================

class Config:
    """系统配置类"""
    VERSION = "3.0"
    MAX_WORKERS = 4
    CHUNK_SIZE = {'x': 512, 'y': 512}
    DEFAULT_DPI = 150
    NODATA_VALUE = -9999.0
    MK_SIGNIFICANCE = 0.05
    BFAST_THRESHOLD = 2.0
    STL_DEFAULT_PERIOD = 12
    ANIMATION_FPS = 2

    # 新增配置
    SMOOTH_WINDOW = 5
    SMOOTH_POLYORDER = 2
    OUTLIER_THRESHOLD = 3.0
    CLUSTER_DEFAULT = 5

    # 颜色方案
    COLORMAPS = {
        'diverging': 'RdBu_r',
        'sequential': 'viridis',
        'trend': 'RdYlGn',
        'cluster': 'tab10'
    }


# ==================== 工具类 ====================

class ProgressTracker:
    """进度跟踪器"""

    def __init__(self, total_steps=100):
        self.total_steps = total_steps
        self.current_step = 0
        self.callbacks = []
        self.is_cancelled = False

    def update(self, step_name="", progress=None):
        """更新进度"""
        if progress is not None:
            self.current_step = progress
        else:
            self.current_step += 1

        percentage = min(100, (self.current_step / self.total_steps) * 100)

        for callback in self.callbacks:
            try:
                callback(step_name, percentage)
            except Exception as e:
                logger.error(f"Progress callback error: {e}")

    def add_callback(self, callback):
        """添加回调函数"""
        self.callbacks.append(callback)

    def cancel(self):
        """取消操作"""
        self.is_cancelled = True
        logger.info("Operation cancelled by user")

    def reset(self):
        """重置"""
        self.current_step = 0
        self.is_cancelled = False


class TimeExtractor:
    """时间信息提取器"""

    @staticmethod
    def extract_time(filename):
        """从文件名中提取时间信息"""
        # 年-儒略日格式: NDVI_2000_123.tif
        m = re.search(r'(19\d{2}|20\d{2})_(\d{3})', filename)
        if m:
            year = int(m.group(1))
            day_of_year = int(m.group(2))
            try:
                date = datetime.datetime(year, 1, 1) + datetime.timedelta(days=day_of_year - 1)
                return date
            except:
                return datetime.datetime(year, 1, 1)

        # 年-月格式: NDVI_2000_01.tif
        m = re.search(r'(19\d{2}|20\d{2})_(\d{1,2})', filename)
        if m:
            year = int(m.group(1))
            month = int(m.group(2))
            if 1 <= month <= 12:
                return datetime.datetime(year, month, 1)

        # 年月连续格式: NDVI_200001.tif
        m = re.search(r'(19\d{2}|20\d{2})(\d{2})', filename)
        if m:
            year = int(m.group(1))
            month = int(m.group(2))
            if 1 <= month <= 12:
                return datetime.datetime(year, month, 1)

        # 仅年份格式: NDVI_2000.tif
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

        return None

    @staticmethod
    def convert_to_years(times):
        """将时间数组转换为年份"""
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


# ==================== 数据预处理模块 ====================

class DataPreprocessor:
    """数据预处理器 - 新增功能"""

    @staticmethod
    def smooth_savgol(stack: xr.DataArray, window_length=None, polyorder=None,
                      progress_tracker=None):
        """Savitzky-Golay平滑滤波"""
        if window_length is None:
            window_length = Config.SMOOTH_WINDOW
        if polyorder is None:
            polyorder = Config.SMOOTH_POLYORDER

        data = stack.values
        n_time, ny, nx = data.shape
        smoothed = np.full_like(data, np.nan)

        total = ny * nx
        processed = 0

        logger.info(f"Starting Savitzky-Golay smoothing (window={window_length}, poly={polyorder})")

        for i in range(ny):
            if progress_tracker and progress_tracker.is_cancelled:
                break
            for j in range(nx):
                ts = data[:, i, j]
                mask = ~np.isnan(ts)

                if np.sum(mask) >= window_length:
                    try:
                        # 对有效数据进行平滑
                        valid_indices = np.where(mask)[0]
                        valid_ts = ts[mask]

                        if len(valid_ts) >= window_length:
                            smoothed_ts = savgol_filter(valid_ts, window_length, polyorder)
                            smoothed[valid_indices, i, j] = smoothed_ts
                        else:
                            smoothed[:, i, j] = ts
                    except Exception as e:
                        smoothed[:, i, j] = ts
                else:
                    smoothed[:, i, j] = ts

                processed += 1
                if progress_tracker and processed % 1000 == 0:
                    progress_tracker.update("数据平滑中", (processed / total) * 100)

        result = stack.copy(deep=True)
        result.values = smoothed
        logger.info(f"Smoothing completed: {processed} pixels processed")
        return result

    @staticmethod
    def smooth_moving_average(stack: xr.DataArray, window=3, progress_tracker=None):
        """移动平均平滑"""
        data = stack.values
        n_time, ny, nx = data.shape
        smoothed = np.full_like(data, np.nan)

        total = ny * nx
        processed = 0

        for i in range(ny):
            if progress_tracker and progress_tracker.is_cancelled:
                break
            for j in range(nx):
                ts = data[:, i, j]
                mask = ~np.isnan(ts)

                if np.sum(mask) >= window:
                    for t in range(n_time):
                        start = max(0, t - window // 2)
                        end = min(n_time, t + window // 2 + 1)
                        window_data = ts[start:end]
                        smoothed[t, i, j] = np.nanmean(window_data)
                else:
                    smoothed[:, i, j] = ts

                processed += 1
                if progress_tracker and processed % 1000 == 0:
                    progress_tracker.update("移动平均中", (processed / total) * 100)

        result = stack.copy(deep=True)
        result.values = smoothed
        return result

    @staticmethod
    def detect_outliers(stack: xr.DataArray, method='zscore', threshold=None,
                        progress_tracker=None):
        """异常值检测
        method: 'zscore' 或 'iqr'
        """
        if threshold is None:
            threshold = Config.OUTLIER_THRESHOLD

        data = stack.values
        n_time, ny, nx = data.shape
        outlier_mask = np.zeros_like(data, dtype=bool)

        total = ny * nx
        processed = 0
        outlier_count = 0

        logger.info(f"Starting outlier detection (method={method}, threshold={threshold})")

        for i in range(ny):
            if progress_tracker and progress_tracker.is_cancelled:
                break
            for j in range(nx):
                ts = data[:, i, j]
                mask = ~np.isnan(ts)

                if np.sum(mask) >= 3:
                    valid_ts = ts[mask]
                    valid_indices = np.where(mask)[0]

                    if method == 'zscore':
                        # Z-score方法
                        mean = np.mean(valid_ts)
                        std = np.std(valid_ts)
                        if std > 0:
                            z_scores = np.abs((valid_ts - mean) / std)
                            outliers = z_scores > threshold
                        else:
                            outliers = np.zeros(len(valid_ts), dtype=bool)

                    elif method == 'iqr':
                        # IQR方法
                        q1, q3 = np.percentile(valid_ts, [25, 75])
                        iqr = q3 - q1
                        lower = q1 - threshold * iqr
                        upper = q3 + threshold * iqr
                        outliers = (valid_ts < lower) | (valid_ts > upper)

                    outlier_mask[valid_indices, i, j] = outliers
                    outlier_count += np.sum(outliers)

                processed += 1
                if progress_tracker and processed % 1000 == 0:
                    progress_tracker.update("异常值检测中", (processed / total) * 100)

        logger.info(f"Outlier detection completed: {outlier_count} outliers found")
        return outlier_mask

    @staticmethod
    def remove_outliers(stack: xr.DataArray, outlier_mask, replace_method='interpolate'):
        """移除异常值
        replace_method: 'nan', 'interpolate', 'mean'
        """
        data = stack.values.copy()
        n_time, ny, nx = data.shape

        if replace_method == 'nan':
            data[outlier_mask] = np.nan

        elif replace_method == 'interpolate':
            for i in range(ny):
                for j in range(nx):
                    ts = data[:, i, j]
                    outliers = outlier_mask[:, i, j]

                    if np.any(outliers):
                        # 将异常值设为NaN后插值
                        ts[outliers] = np.nan
                        mask = ~np.isnan(ts)

                        if np.sum(mask) >= 2:
                            x = np.arange(n_time)
                            ts_interp = np.interp(x, x[mask], ts[mask])
                            data[:, i, j] = ts_interp

        elif replace_method == 'mean':
            for i in range(ny):
                for j in range(nx):
                    ts = data[:, i, j]
                    outliers = outlier_mask[:, i, j]

                    if np.any(outliers):
                        valid_mean = np.nanmean(ts[~outliers])
                        data[outliers, i, j] = valid_mean

        result = stack.copy(deep=True)
        result.values = data
        return result

    @staticmethod
    def interpolate_gaps(stack: xr.DataArray, method='linear', progress_tracker=None):
        """插值填补缺失值
        method: 'linear', 'cubic', 'nearest'
        """
        data = stack.values
        n_time, ny, nx = data.shape
        interpolated = data.copy()

        total = ny * nx
        processed = 0

        logger.info(f"Starting gap interpolation (method={method})")

        for i in range(ny):
            if progress_tracker and progress_tracker.is_cancelled:
                break
            for j in range(nx):
                ts = data[:, i, j]
                mask = ~np.isnan(ts)

                if np.sum(mask) >= 2 and np.sum(~mask) > 0:
                    x = np.arange(n_time)

                    try:
                        if method == 'linear':
                            interpolated[:, i, j] = np.interp(x, x[mask], ts[mask])
                        elif method == 'cubic' and np.sum(mask) >= 4:
                            cs = CubicSpline(x[mask], ts[mask])
                            interpolated[:, i, j] = cs(x)
                        elif method == 'nearest':
                            # 最近邻插值
                            for t in range(n_time):
                                if not mask[t]:
                                    valid_indices = x[mask]
                                    nearest_idx = valid_indices[np.argmin(np.abs(valid_indices - t))]
                                    interpolated[t, i, j] = ts[nearest_idx]
                    except Exception as e:
                        logger.warning(f"Interpolation failed at ({i},{j}): {e}")
                        continue

                processed += 1
                if progress_tracker and processed % 1000 == 0:
                    progress_tracker.update("数据插值中", (processed / total) * 100)

        result = stack.copy(deep=True)
        result.values = interpolated
        logger.info(f"Interpolation completed: {processed} pixels processed")
        return result

    @staticmethod
    def spatial_subset(stack: xr.DataArray, bbox):
        """空间裁剪
        bbox: (xmin, ymin, xmax, ymax) 或 (row_min, row_max, col_min, col_max)
        """
        try:
            if hasattr(stack, 'x') and hasattr(stack, 'y'):
                # 使用地理坐标裁剪
                xmin, ymin, xmax, ymax = bbox
                subset = stack.sel(x=slice(xmin, xmax), y=slice(ymax, ymin))
            else:
                # 使用索引裁剪
                row_min, row_max, col_min, col_max = bbox
                subset = stack.isel(y=slice(row_min, row_max), x=slice(col_min, col_max))

            logger.info(f"Spatial subset created: shape {subset.shape}")
            return subset
        except Exception as e:
            logger.error(f"Spatial subset failed: {e}")
            return stack

    @staticmethod
    def temporal_subset(stack: xr.DataArray, start_date, end_date):
        """时间子集提取"""
        try:
            subset = stack.sel(time=slice(start_date, end_date))
            logger.info(f"Temporal subset created: {len(subset.time)} time steps")
            return subset
        except Exception as e:
            logger.error(f"Temporal subset failed: {e}")
            return stack


# ==================== 增强的分析算法 ====================

class TrendAnalyzer:
    """趋势分析器"""

    @staticmethod
    def theil_sen(stack: xr.DataArray, progress_tracker=None):
        """Theil-Sen趋势分析"""
        data = stack.values
        time_idx = np.arange(data.shape[0])
        ny, nx = data.shape[1], data.shape[2]

        slope = np.full((ny, nx), np.nan, dtype=np.float32)
        intercept = np.full((ny, nx), np.nan, dtype=np.float32)
        nan_mask = np.all(np.isnan(data), axis=0)

        total_pixels = ny * nx
        processed = 0

        for i in range(ny):
            if progress_tracker and progress_tracker.is_cancelled:
                break

            for j in range(nx):
                if nan_mask[i, j]:
                    continue

                ts = data[:, i, j]
                if not np.isnan(ts).all():
                    try:
                        res = stats.theilslopes(ts, time_idx)
                        slope[i, j] = res[0]
                        intercept[i, j] = res[1]
                    except:
                        continue

                processed += 1
                if progress_tracker and processed % 1000 == 0:
                    progress_tracker.update("Theil-Sen分析中", (processed / total_pixels) * 100)

        coords = {"y": stack.y, "x": stack.x}
        return (
            xr.DataArray(slope, dims=("y", "x"), coords=coords),
            xr.DataArray(intercept, dims=("y", "x"), coords=coords)
        )

    @staticmethod
    def mann_kendall(stack: xr.DataArray, significance=None, progress_tracker=None):
        """Mann-Kendall趋势检验"""
        if significance is None:
            significance = Config.MK_SIGNIFICANCE

        from scipy.stats import kendalltau

        data = stack.values
        ny, nx = data.shape[1], data.shape[2]
        out = np.full((ny, nx), np.nan, dtype=np.float32)
        time_idx = np.arange(data.shape[0])
        nan_mask = np.all(np.isnan(data), axis=0)

        total_pixels = ny * nx
        processed = 0

        for i in range(ny):
            if progress_tracker and progress_tracker.is_cancelled:
                break

            for j in range(nx):
                if nan_mask[i, j]:
                    continue

                ts = data[:, i, j]
                mask = ~np.isnan(ts)

                if np.sum(mask) < 3:
                    continue

                try:
                    valid_ts = ts[mask]
                    valid_time = time_idx[mask]
                    tau, p_value = kendalltau(valid_time, valid_ts)

                    if not np.isnan(p_value) and not np.isnan(tau):
                        if p_value < significance:
                            out[i, j] = 1.0 if tau > 0 else -1.0
                        else:
                            out[i, j] = 0.0
                except:
                    continue

                processed += 1
                if progress_tracker and processed % 1000 == 0:
                    progress_tracker.update("Mann-Kendall检验中", (processed / total_pixels) * 100)

        return xr.DataArray(out, dims=("y", "x"), coords={"y": stack.y, "x": stack.x})


class BreakpointDetector:
    """突变点检测器"""

    @staticmethod
    def bfast(stack: xr.DataArray, threshold=None, progress_tracker=None):
        """BFAST突变检测"""
        if threshold is None:
            threshold = Config.BFAST_THRESHOLD

        times = stack["time"].values
        years = TimeExtractor.convert_to_years(times)

        data = stack.values
        n_time = data.shape[0]
        ny, nx = data.shape[1], data.shape[2]
        break_data = np.full((ny, nx), np.nan, dtype=np.float32)
        nan_mask = np.all(np.isnan(data), axis=0)

        total_pixels = ny * nx
        processed = 0

        for i in range(ny):
            if progress_tracker and progress_tracker.is_cancelled:
                break

            for j in range(nx):
                if nan_mask[i, j]:
                    continue

                ts = data[:, i, j]
                mask = ~np.isnan(ts)

                if np.sum(mask) < 4:
                    continue

                try:
                    x = np.arange(n_time)
                    coeffs = np.polyfit(x[mask], ts[mask], 1)
                    trend = np.polyval(coeffs, x)
                    residuals = ts - trend
                    residual_std = np.nanstd(residuals)

                    if residual_std > 0:
                        z_scores = np.abs(residuals) / residual_std
                        break_points = np.where(z_scores > threshold)[0]

                        if len(break_points) > 0:
                            break_idx = break_points[0]
                            break_data[i, j] = float(years[break_idx])
                except:
                    continue

                processed += 1
                if progress_tracker and processed % 1000 == 0:
                    progress_tracker.update("BFAST突变检测中", (processed / total_pixels) * 100)

        result = xr.DataArray(break_data, dims=("y", "x"), coords={"y": stack.y, "x": stack.x})
        return BreakpointDetector._fix_results(result)

    @staticmethod
    def _fix_results(break_da):
        """修复BFAST结果"""
        break_values = break_da.values
        break_values_fixed = np.full_like(break_values, np.nan)
        current_year = datetime.datetime.now().year

        for i in range(break_values.shape[0]):
            for j in range(break_values.shape[1]):
                val = break_values[i, j]
                if not np.isnan(val):
                    if val > 1e18:
                        try:
                            dt = pd.to_datetime(val)
                            if 1900 <= dt.year <= current_year + 1:
                                break_values_fixed[i, j] = dt.year
                        except:
                            pass
                    elif 1900 <= val <= current_year + 1:
                        break_values_fixed[i, j] = val

        return xr.DataArray(break_values_fixed, dims=break_da.dims, coords=break_da.coords)


class FrequencyAnalyzer:
    """频率分析器"""

    @staticmethod
    def fft(stack: xr.DataArray, progress_tracker=None):
        """FFT周期分析"""
        data = stack.values
        n = data.shape[0]
        ny, nx = data.shape[1], data.shape[2]

        amp = np.full((ny, nx), np.nan, dtype=np.float32)
        period = np.full((ny, nx), np.nan, dtype=np.float32)
        nan_mask = np.all(np.isnan(data), axis=0)

        total_pixels = ny * nx
        processed = 0

        for i in range(ny):
            if progress_tracker and progress_tracker.is_cancelled:
                break

            for j in range(nx):
                if nan_mask[i, j]:
                    continue

                ts = data[:, i, j]
                if np.isnan(ts).all():
                    continue

                try:
                    y = ts - np.nanmean(ts)
                    yf = fftpack.fft(y)
                    xf = fftpack.fftfreq(n, d=1)

                    half = n // 2
                    power = np.abs(yf[:half])
                    power[0] = 0

                    if power.size > 1:
                        idx = np.argmax(power[1:]) + 1
                        amp[i, j] = float(power[idx])

                        freq = xf[idx]
                        if freq != 0:
                            period[i, j] = float(1.0 / freq)
                except:
                    continue

                processed += 1
                if progress_tracker and processed % 1000 == 0:
                    progress_tracker.update("FFT分析中", (processed / total_pixels) * 100)

        coords = {"y": stack.y, "x": stack.x}
        return (
            xr.DataArray(amp, dims=("y", "x"), coords=coords),
            xr.DataArray(period, dims=("y", "x"), coords=coords)
        )


class STLDecomposer:
    """STL分解器"""

    @staticmethod
    def decompose(stack: xr.DataArray, period=None, progress_tracker=None):
        """STL分解"""
        if period is None:
            period = Config.STL_DEFAULT_PERIOD

        data = stack.values
        n, ny, nx = data.shape

        trend_mean = np.full((ny, nx), np.nan, dtype=np.float32)
        seasonal_mean = np.full((ny, nx), np.nan, dtype=np.float32)
        resid_std = np.full((ny, nx), np.nan, dtype=np.float32)
        nan_mask = np.all(np.isnan(data), axis=0)

        total_pixels = ny * nx
        processed = 0

        for i in range(ny):
            if progress_tracker and progress_tracker.is_cancelled:
                break

            for j in range(nx):
                if nan_mask[i, j]:
                    continue

                ts = data[:, i, j]
                if np.sum(~np.isnan(ts)) < period * 2:
                    continue

                try:
                    ts_filled = ts.copy()
                    mask = ~np.isnan(ts)

                    if not np.all(mask):
                        x = np.arange(n)
                        ts_filled = np.interp(x, x[mask], ts[mask])

                    stl = STL(ts_filled, period=period, robust=True)
                    res = stl.fit()

                    trend_mean[i, j] = np.mean(res.trend)
                    seasonal_mean[i, j] = np.mean(res.seasonal)
                    resid_std[i, j] = np.std(res.resid)
                except:
                    continue

                processed += 1
                if progress_tracker and processed % 1000 == 0:
                    progress_tracker.update("STL分解中", (processed / total_pixels) * 100)

        coords = {"y": stack.y, "x": stack.x}
        return (
            xr.DataArray(trend_mean, dims=("y", "x"), coords=coords),
            xr.DataArray(seasonal_mean, dims=("y", "x"), coords=coords),
            xr.DataArray(resid_std, dims=("y", "x"), coords=coords)
        )


# ==================== 新增功能类 ====================

class AnimationGenerator:
    """动画生成器 - 新增功能"""

    @staticmethod
    def create_timeseries_animation(stack: xr.DataArray, output_path,
                                    fps=None, cmap='viridis', dpi=100,
                                    title_template="时间: {time}",
                                    progress_callback=None):
        """生成时序动画

        参数:
            stack: 时间序列数据
            output_path: 输出路径 (.gif 或 .mp4)
            fps: 帧率
            cmap: 颜色映射
            dpi: 分辨率
            title_template: 标题模板
            progress_callback: 进度回调函数
        """
        if fps is None:
            fps = Config.ANIMATION_FPS

        logger.info(f"Creating animation: {output_path}")

        times = stack.time.values
        n_frames = len(times)

        # 计算全局vmin/vmax以保持颜色一致性
        vmin = float(np.nanpercentile(stack.values, 2))
        vmax = float(np.nanpercentile(stack.values, 98))

        fig, ax = plt.subplots(figsize=(10, 8))

        def init():
            ax.clear()
            return []

        def animate(frame):
            ax.clear()
            data = stack.isel(time=frame).values

            im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)

            # 格式化时间标签
            try:
                time_str = pd.to_datetime(str(times[frame])).strftime('%Y-%m-%d')
            except:
                time_str = str(times[frame])

            title = title_template.format(time=time_str, frame=frame + 1, total=n_frames)
            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.axis('off')

            # 添加色标
            if frame == 0:
                cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                cbar.set_label('值', rotation=270, labelpad=15)

            if progress_callback:
                progress_callback("生成动画帧", ((frame + 1) / n_frames) * 100)

            return [im]

        try:
            anim = animation.FuncAnimation(
                fig, animate, init_func=init,
                frames=n_frames, interval=1000 / fps, blit=False
            )

            # 保存动画
            if output_path.endswith('.gif'):
                writer = animation.PillowWriter(fps=fps)
                anim.save(output_path, writer=writer, dpi=dpi)
            elif output_path.endswith('.mp4'):
                writer = animation.FFMpegWriter(fps=fps, bitrate=1800)
                anim.save(output_path, writer=writer, dpi=dpi)
            else:
                raise ValueError("输出格式必须是 .gif 或 .mp4")

            plt.close(fig)
            logger.info(f"Animation saved: {output_path}")
            return output_path

        except Exception as e:
            plt.close(fig)
            logger.error(f"Animation creation failed: {e}")
            raise


class TimeSeriesClusterer:
    """时间序列聚类分析 - 新增功能"""

    @staticmethod
    def kmeans_clustering(stack: xr.DataArray, n_clusters=None,
                          max_iter=100, random_state=42,
                          progress_tracker=None):
        """K-means聚类

        返回:
            cluster_map: 聚类标签图
            centers: 聚类中心时间序列
            metrics: 聚类质量指标
        """
        if n_clusters is None:
            n_clusters = Config.CLUSTER_DEFAULT

        logger.info(f"Starting K-means clustering (n_clusters={n_clusters})")

        data = stack.values
        n_time, ny, nx = data.shape

        if progress_tracker:
            progress_tracker.update("准备聚类数据", 5)

        # 重塑为 (n_pixels, n_time)
        reshaped = data.transpose(1, 2, 0).reshape(-1, n_time)

        # 移除全NaN的像元
        valid_mask = ~np.all(np.isnan(reshaped), axis=1)
        valid_data = reshaped[valid_mask]

        logger.info(f"Valid pixels for clustering: {len(valid_data)}")

        if progress_tracker:
            progress_tracker.update("数据插值", 15)

        # 对有NaN的序列进行插值
        for i in range(len(valid_data)):
            ts = valid_data[i]
            if np.any(np.isnan(ts)):
                mask = ~np.isnan(ts)
                if np.sum(mask) >= 2:
                    x = np.arange(n_time)
                    valid_data[i] = np.interp(x, x[mask], ts[mask])
                else:
                    valid_data[i] = np.nanmean(ts)

        if progress_tracker:
            progress_tracker.update("数据标准化", 25)

        # 标准化
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(valid_data)

        if progress_tracker:
            progress_tracker.update("K-means聚类计算", 35)

        # 聚类
        kmeans = KMeans(
            n_clusters=n_clusters,
            max_iter=max_iter,
            random_state=random_state,
            n_init=10
        )
        labels = kmeans.fit_predict(scaled_data)

        if progress_tracker:
            progress_tracker.update("生成结果", 80)

        # 重建为空间形状
        full_labels = np.full(ny * nx, -1, dtype=int)
        full_labels[valid_mask] = labels
        cluster_map = full_labels.reshape(ny, nx)

        # 计算聚类中心（原始尺度）
        centers = scaler.inverse_transform(kmeans.cluster_centers_)

        # 计算聚类质量指标
        from sklearn.metrics import silhouette_score, calinski_harabasz_score

        metrics = {
            'inertia': kmeans.inertia_,
            'silhouette': silhouette_score(scaled_data, labels),
            'calinski_harabasz': calinski_harabasz_score(scaled_data, labels)
        }

        result = xr.DataArray(
            cluster_map,
            dims=('y', 'x'),
            coords={'y': stack.y, 'x': stack.x}
        )

        logger.info(f"Clustering completed. Metrics: {metrics}")

        if progress_tracker:
            progress_tracker.update("聚类完成", 100)

        return result, centers, metrics

    @staticmethod
    def hierarchical_clustering(stack: xr.DataArray, n_clusters=None,
                                linkage='ward', progress_tracker=None):
        """层次聚类"""
        if n_clusters is None:
            n_clusters = Config.CLUSTER_DEFAULT

        logger.info(f"Starting hierarchical clustering (n_clusters={n_clusters}, linkage={linkage})")

        data = stack.values
        n_time, ny, nx = data.shape

        reshaped = data.transpose(1, 2, 0).reshape(-1, n_time)
        valid_mask = ~np.all(np.isnan(reshaped), axis=1)
        valid_data = reshaped[valid_mask]

        # 插值处理NaN
        for i in range(len(valid_data)):
            ts = valid_data[i]
            if np.any(np.isnan(ts)):
                mask = ~np.isnan(ts)
                if np.sum(mask) >= 2:
                    x = np.arange(n_time)
                    valid_data[i] = np.interp(x, x[mask], ts[mask])

        # 标准化和聚类
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(valid_data)

        if progress_tracker:
            progress_tracker.update("层次聚类计算", 50)

        clustering = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage=linkage
        )
        labels = clustering.fit_predict(scaled_data)

        # 重建空间形状
        full_labels = np.full(ny * nx, -1, dtype=int)
        full_labels[valid_mask] = labels
        cluster_map = full_labels.reshape(ny, nx)

        result = xr.DataArray(
            cluster_map,
            dims=('y', 'x'),
            coords={'y': stack.y, 'x': stack.x}
        )

        logger.info("Hierarchical clustering completed")
        return result


# ==================== 数据导出类 ====================

class DataExporter:
    """数据导出器"""

    @staticmethod
    def to_geotiff(data_array, reference_stack=None, nodata=None):
        """转换为GeoTIFF字节数据"""
        if nodata is None:
            nodata = Config.NODATA_VALUE

        # 转换为2D数组
        arr2d = DataExporter._to_2d_array(data_array)
        arr2d = np.where(np.isnan(arr2d), nodata, arr2d).astype(np.float32)

        try:
            # 获取空间参考信息
            crs, transform = DataExporter._get_spatial_reference(data_array, reference_stack)

            # 创建配置
            profile = {
                'driver': 'GTiff',
                'dtype': rasterio.float32,
                'count': 1,
                'height': arr2d.shape[0],
                'width': arr2d.shape[1],
                'compress': 'lzw',
                'nodata': nodata
            }

            if crs is not None:
                profile['crs'] = crs
            if transform is not None:
                profile['transform'] = transform
            else:
                profile['transform'] = from_origin(0, arr2d.shape[0], 1, 1)

            # 写入内存文件
            memfile = MemoryFile()
            with memfile.open(**profile) as dst:
                dst.write(arr2d, 1)

            data = memfile.read()
            memfile.close()
            return data

        except Exception as e:
            logger.error(f"GeoTIFF generation failed: {e}")
            return DataExporter._create_simple_tiff(arr2d, nodata)

    @staticmethod
    def to_csv(data_array, output_path, include_coords=True):
        """导出为CSV格式"""
        try:
            df = data_array.to_dataframe(name='value')
            if not include_coords:
                df = df.reset_index(drop=True)
            df.to_csv(output_path, index=include_coords)
            logger.info(f"CSV exported: {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"CSV export failed: {e}")
            raise

    @staticmethod
    def _to_2d_array(da):
        """转换为2D数组"""
        if isinstance(da, xr.DataArray):
            if "time" in da.dims and "y" in da.dims and "x" in da.dims:
                return np.nanmean(da.values, axis=0)
            elif "y" in da.dims and "x" in da.dims:
                return da.values
            else:
                vals = da.values
                if vals.ndim >= 2:
                    return np.nanmean(vals, axis=tuple(range(vals.ndim - 2)))
                return vals
        return np.array(da)

    @staticmethod
    def _get_spatial_reference(data_array, reference_stack):
        """获取空间参考信息"""
        crs = None
        transform = None

        if hasattr(data_array, 'rio') and data_array.rio.crs is not None:
            crs = data_array.rio.crs
            transform = data_array.rio.transform()

        if crs is None and reference_stack is not None:
            try:
                ref_da = reference_stack.isel(time=0) if 'time' in reference_stack.dims else reference_stack
                if hasattr(ref_da, 'rio') and ref_da.rio.crs is not None:
                    crs = ref_da.rio.crs
                    transform = ref_da.rio.transform()
            except:
                pass

        if transform is None:
            transform = DataExporter._infer_transform(data_array)

        return crs, transform

    @staticmethod
    def _infer_transform(da):
        """从坐标推断变换"""
        try:
            if hasattr(da, 'x') and hasattr(da, 'y'):
                if len(da.x) > 1 and len(da.y) > 1:
                    x_res = float(da.x[1] - da.x[0])
                    y_res = float(da.y[0] - da.y[1])
                    return from_origin(
                        float(da.x[0]) - x_res / 2,
                        float(da.y[0]) + y_res / 2,
                        x_res,
                        y_res
                    )
        except:
            pass
        return None

    @staticmethod
    def _create_simple_tiff(arr2d, nodata):
        """创建简单TIFF"""
        try:
            profile = {
                'driver': 'GTiff',
                'dtype': rasterio.float32,
                'count': 1,
                'height': arr2d.shape[0],
                'width': arr2d.shape[1],
                'transform': from_origin(0, arr2d.shape[0], 1, 1),
                'compress': 'lzw',
                'nodata': nodata
            }

            memfile = MemoryFile()
            with memfile.open(**profile) as dst:
                dst.write(arr2d, 1)

            data = memfile.read()
            memfile.close()
            return data
        except Exception as e:
            logger.error(f"Simple TIFF creation failed: {e}")
            return b''


# ==================== 可视化增强类 ====================

class Visualizer:
    """增强的可视化器"""

    @staticmethod
    def create_result_figure(data_array, title, cmap=None, vmin=None, vmax=None,
                             add_stats=True, figsize=(10, 8)):
        """创建结果图表"""
        if cmap is None:
            cmap = Config.COLORMAPS['sequential']

        fig, ax = plt.subplots(figsize=figsize)

        data = Visualizer._prepare_data(data_array)

        # 自动计算vmin/vmax（使用百分位数避免极值影响）
        if vmin is None:
            vmin = np.nanpercentile(data, 2)
        if vmax is None:
            vmax = np.nanpercentile(data, 98)

        im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=14, fontweight='bold', pad=15)

        # 添加色标
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=10)

        # 添加统计信息
        if add_stats:
            valid_data = data[~np.isnan(data)]
            if len(valid_data) > 0:
                stats_text = f'Min: {np.min(valid_data):.3f}\nMax: {np.max(valid_data):.3f}\nMean: {np.mean(valid_data):.3f}'
                ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                        verticalalignment='top', bbox=dict(boxstyle='round',
                                                           facecolor='white', alpha=0.8), fontsize=9)

        ax.axis('off')
        plt.tight_layout()
        return fig

    @staticmethod
    def create_multi_panel_figure(data_arrays, titles, cmaps=None,
                                  figsize=(18, 5), ncols=None):
        """创建多面板图表"""
        n_panels = len(data_arrays)

        if ncols is None:
            ncols = min(3, n_panels)
        nrows = int(np.ceil(n_panels / ncols))

        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)

        if n_panels == 1:
            axes = np.array([axes])
        axes = axes.flatten()

        if cmaps is None:
            cmaps = [Config.COLORMAPS['sequential']] * n_panels

        for i, (data_array, title, cmap) in enumerate(zip(data_arrays, titles, cmaps)):
            data = Visualizer._prepare_data(data_array)

            vmin = np.nanpercentile(data, 2)
            vmax = np.nanpercentile(data, 98)

            im = axes[i].imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
            axes[i].set_title(title, fontsize=12, fontweight='bold')
            plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
            axes[i].axis('off')

        # 隐藏多余的子图
        for i in range(n_panels, len(axes)):
            axes[i].axis('off')

        plt.tight_layout()
        return fig

    @staticmethod
    def create_pixel_analysis_figure(stack, row, col, period=12):
        """创建像元分析图表"""
        series = stack[:, row, col].values
        times = stack["time"].values
        time_labels = Visualizer._format_time_labels(times)

        fig = plt.figure(figsize=(15, 10))
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

        fig.suptitle(f'像元 ({int(row)}, {int(col)}) 时序分析',
                     fontsize=16, fontweight='bold')

        # 1. 原始时序
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(time_labels, series, 'o-', linewidth=2, markersize=5,
                 color='#2E86AB', alpha=0.7, label='原始数据')
        ax1.set_title("原始时序", fontsize=12, fontweight='bold')
        ax1.set_ylabel("值", fontsize=10)
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.tick_params(axis='x', rotation=45)
        ax1.legend()

        # 2. 趋势分析
        ax2 = fig.add_subplot(gs[1, 0])
        Visualizer._add_trend_plot(ax2, series, time_labels)

        # 3. 统计信息
        ax3 = fig.add_subplot(gs[1, 1])
        Visualizer._add_statistics_plot(ax3, series)

        # 4. STL趋势分量
        ax4 = fig.add_subplot(gs[2, 0])

        # 5. STL季节分量
        ax5 = fig.add_subplot(gs[2, 1])

        Visualizer._add_stl_plots(ax4, ax5, series, time_labels, period)

        return fig

    @staticmethod
    def create_cluster_visualization(cluster_map, centers, times, n_samples=5):
        """创建聚类结果可视化"""
        n_clusters = len(centers)

        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], hspace=0.3, wspace=0.3)

        # 1. 聚类地图
        ax1 = fig.add_subplot(gs[0, :])
        im = ax1.imshow(cluster_map, cmap=Config.COLORMAPS['cluster'],
                        vmin=-0.5, vmax=n_clusters - 0.5)
        ax1.set_title('聚类结果地图', fontsize=14, fontweight='bold')
        cbar = plt.colorbar(im, ax=ax1, ticks=range(n_clusters))
        cbar.set_label('聚类标签', rotation=270, labelpad=15)
        ax1.axis('off')

        # 2. 聚类中心时序
        ax2 = fig.add_subplot(gs[1, 0])
        time_labels = Visualizer._format_time_labels(times)

        colors = plt.cm.tab10(np.linspace(0, 1, n_clusters))
        for i, (center, color) in enumerate(zip(centers, colors)):
            ax2.plot(time_labels, center, 'o-', linewidth=2,
                     color=color, label=f'聚类 {i}', alpha=0.8)

        ax2.set_title('聚类中心时序', fontsize=12, fontweight='bold')
        ax2.set_xlabel('时间')
        ax2.set_ylabel('值')
        ax2.legend(loc='best', fontsize=9)
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(axis='x', rotation=45)

        # 3. 聚类统计
        ax3 = fig.add_subplot(gs[1, 1])
        cluster_counts = []
        cluster_labels = []

        for i in range(n_clusters):
            count = np.sum(cluster_map == i)
            cluster_counts.append(count)
            cluster_labels.append(f'聚类 {i}')

        ax3.barh(cluster_labels, cluster_counts, color=colors, alpha=0.7)
        ax3.set_xlabel('像元数量')
        ax3.set_title('聚类分布统计', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='x')

        for i, count in enumerate(cluster_counts):
            ax3.text(count, i, f' {count}', va='center', fontsize=9)

        plt.tight_layout()
        return fig

    @staticmethod
    def _prepare_data(data_array):
        """准备显示数据"""
        if isinstance(data_array, xr.DataArray):
            if "time" in data_array.dims:
                return np.nanmean(data_array.values, axis=0)
            return data_array.values
        return np.array(data_array)

    @staticmethod
    def _format_time_labels(times):
        """格式化时间标签"""
        labels = []
        for t in times:
            if isinstance(t, np.datetime64):
                labels.append(np.datetime_as_string(t, unit='D'))
            else:
                labels.append(str(t))
        return labels

    @staticmethod
    def _add_trend_plot(ax, series, time_labels):
        """添加趋势图"""
        valid_mask = ~np.isnan(series)
        if np.sum(valid_mask) >= 3:
            x = np.arange(len(series))
            valid_x = x[valid_mask]
            valid_series = series[valid_mask]

            if len(valid_x) >= 2:
                coeffs = np.polyfit(valid_x, valid_series, 1)
                trend_line = np.polyval(coeffs, x)

                ax.plot(time_labels, series, 'o-', alpha=0.5,
                        color='#2E86AB', label='原始数据', markersize=4)
                ax.plot(time_labels, trend_line, '--', linewidth=2,
                        color='#A23B72', label=f'趋势线 (斜率: {coeffs[0]:.4f})')
                ax.set_title("趋势分析", fontsize=12, fontweight='bold')
                ax.set_ylabel("值", fontsize=10)
                ax.legend(loc='best', fontsize=9)
                ax.grid(True, alpha=0.3, linestyle='--')
                ax.tick_params(axis='x', rotation=45)

    @staticmethod
    def _add_statistics_plot(ax, series):
        """添加统计信息图"""
        valid_data = series[~np.isnan(series)]

        if len(valid_data) > 0:
            stats = {
                '最小值': np.min(valid_data),
                '最大值': np.max(valid_data),
                '均值': np.mean(valid_data),
                '中位数': np.median(valid_data),
                '标准差': np.std(valid_data),
                '变异系数': np.std(valid_data) / np.mean(valid_data) if np.mean(valid_data) != 0 else 0
            }

            # 创建表格
            table_data = [[k, f'{v:.4f}'] for k, v in stats.items()]

            ax.axis('tight')
            ax.axis('off')
            table = ax.table(cellText=table_data, colLabels=['指标', '值'],
                             cellLoc='left', loc='center',
                             colWidths=[0.5, 0.5])
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 2)

            # 设置表头样式
            for i in range(2):
                table[(0, i)].set_facecolor('#2E86AB')
                table[(0, i)].set_text_props(weight='bold', color='white')

            ax.set_title("统计信息", fontsize=12, fontweight='bold')

    @staticmethod
    def _add_stl_plots(ax3, ax4, series, time_labels, period):
        """添加STL分解图"""
        try:
            valid_mask = ~np.isnan(series)
            if np.sum(valid_mask) >= max(3, period * 2):
                series_filled = series.copy()
                if not np.all(valid_mask):
                    x = np.arange(len(series))
                    series_filled = np.interp(x, x[valid_mask], series[valid_mask])

                stl_result = STL(series_filled, period=period, robust=True).fit()

                ax3.plot(time_labels, stl_result.trend, linewidth=2,
                         color='#F18F01', label='趋势分量')
                ax3.fill_between(range(len(time_labels)), stl_result.trend,
                                 alpha=0.3, color='#F18F01')
                ax3.set_title("STL - 趋势分量", fontsize=12, fontweight='bold')
                ax3.set_xlabel("时间", fontsize=10)
                ax3.set_ylabel("值", fontsize=10)
                ax3.grid(True, alpha=0.3, linestyle='--')
                ax3.tick_params(axis='x', rotation=45)
                ax3.legend(fontsize=9)

                ax4.plot(time_labels, stl_result.seasonal, linewidth=2,
                         color='#C73E1D', label='季节分量')
                ax4.fill_between(range(len(time_labels)), stl_result.seasonal,
                                 alpha=0.3, color='#C73E1D')
                ax4.set_title("STL - 季节分量", fontsize=12, fontweight='bold')
                ax4.set_xlabel("时间", fontsize=10)
                ax4.set_ylabel("值", fontsize=10)
                ax4.grid(True, alpha=0.3, linestyle='--')
                ax4.tick_params(axis='x', rotation=45)
                ax4.legend(fontsize=9)
        except Exception as e:
            error_msg = f"STL分析失败\n{str(e)[:50]}"
            ax3.text(0.5, 0.5, error_msg, ha='center', va='center',
                     transform=ax3.transAxes, fontsize=10, color='red')
            ax4.text(0.5, 0.5, error_msg, ha='center', va='center',
                     transform=ax4.transAxes, fontsize=10, color='red')


# ==================== 项目管理器 ====================

class ProjectManager:
    """项目管理器 - 新增功能"""

    @staticmethod
    def save_project(data_stack, analysis_results, parameters, file_path):
        """保存完整项目"""
        try:
            project_data = {
                'version': Config.VERSION,
                'timestamp': datetime.datetime.now().isoformat(),
                'data_stack': data_stack,
                'analysis_results': analysis_results,
                'parameters': parameters,
                'metadata': {
                    'n_time': len(data_stack.time) if data_stack is not None else 0,
                    'shape': data_stack.shape if data_stack is not None else None,
                }
            }

            with open(file_path, 'wb') as f:
                pickle.dump(project_data, f, protocol=pickle.HIGHEST_PROTOCOL)

            logger.info(f"Project saved: {file_path}")
            return True

        except Exception as e:
            logger.error(f"Project save failed: {e}")
            raise

    @staticmethod
    def load_project(file_path):
        """加载项目"""
        try:
            with open(file_path, 'rb') as f:
                project_data = pickle.load(f)

            logger.info(f"Project loaded: {file_path}")
            return project_data

        except Exception as e:
            logger.error(f"Project load failed: {e}")
            raise

    @staticmethod
    def export_parameters(parameters, file_path):
        """导出参数配置为JSON"""
        try:
            # 转换不可序列化的对象
            serializable_params = {}
            for key, value in parameters.items():
                if isinstance(value, (str, int, float, bool, list, dict, type(None))):
                    serializable_params[key] = value
                else:
                    serializable_params[key] = str(value)

            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(serializable_params, f, indent=2, ensure_ascii=False)

            logger.info(f"Parameters exported: {file_path}")
            return True

        except Exception as e:
            logger.error(f"Parameter export failed: {e}")
            raise

    @staticmethod
    def import_parameters(file_path):
        """导入参数配置"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                parameters = json.load(f)

            logger.info(f"Parameters imported: {file_path}")
            return parameters

        except Exception as e:
            logger.error(f"Parameter import failed: {e}")
            raise


# ==================== 报告生成器 ====================

class ReportGenerator:
    """分析报告生成器 - 新增功能"""

    @staticmethod
    def generate_text_report(analysis_results, data_info, output_path):
        """生成文本格式报告"""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write("=" * 80 + "\n")
                f.write("时序遥感分析报告\n")
                f.write("=" * 80 + "\n\n")

                # 基本信息
                f.write("【基本信息】\n")
                f.write(f"生成时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"数据时间范围: {data_info.get('time_range', 'N/A')}\n")
                f.write(f"时间序列长度: {data_info.get('n_time', 'N/A')} 期\n")
                f.write(f"空间大小: {data_info.get('ny', 'N/A')} × {data_info.get('nx', 'N/A')} 像元\n\n")

                # 分析结果
                for analysis_name, results in analysis_results.items():
                    f.write("-" * 80 + "\n")
                    f.write(f"【{analysis_name} 分析结果】\n")
                    f.write("-" * 80 + "\n")

                    if isinstance(results, dict):
                        for key, data_array in results.items():
                            stats = ReportGenerator._calculate_statistics(data_array)
                            f.write(f"\n{key}:\n")
                            f.write(f"  最小值: {stats['min']:.6f}\n")
                            f.write(f"  最大值: {stats['max']:.6f}\n")
                            f.write(f"  平均值: {stats['mean']:.6f}\n")
                            f.write(f"  标准差: {stats['std']:.6f}\n")
                            f.write(f"  有效像元数: {stats['valid_count']:,}\n")
                    else:
                        stats = ReportGenerator._calculate_statistics(results)
                        f.write(f"  最小值: {stats['min']:.6f}\n")
                        f.write(f"  最大值: {stats['max']:.6f}\n")
                        f.write(f"  平均值: {stats['mean']:.6f}\n")
                        f.write(f"  标准差: {stats['std']:.6f}\n")
                        f.write(f"  有效像元数: {stats['valid_count']:,}\n")

                    f.write("\n")

                f.write("=" * 80 + "\n")
                f.write("报告结束\n")
                f.write("=" * 80 + "\n")

            logger.info(f"Text report generated: {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"Text report generation failed: {e}")
            raise

    @staticmethod
    def _calculate_statistics(data_array):
        """计算统计信息"""
        values = data_array.values if isinstance(data_array, xr.DataArray) else np.array(data_array)

        # 转为2D
        if values.ndim > 2:
            values = np.nanmean(values, axis=0)

        valid = values[~np.isnan(values)]

        return {
            'min': float(np.min(valid)) if len(valid) > 0 else np.nan,
            'max': float(np.max(valid)) if len(valid) > 0 else np.nan,
            'mean': float(np.mean(valid)) if len(valid) > 0 else np.nan,
            'std': float(np.std(valid)) if len(valid) > 0 else np.nan,
            'valid_count': int(len(valid))
        }


# ==================== 交互式功能 ====================

class InteractiveTools:
    """交互式工具集"""

    @staticmethod
    def create_interactive_map(data_array, parent_window, data_stack=None):
        """创建交互式地图查看器"""
        viewer_window = tb.Toplevel(parent_window)
        viewer_window.title("交互式地图查看器")
        viewer_window.geometry("1200x800")

        # 准备显示数据
        if "time" in data_array.dims:
            display_data = data_array.isel(time=0).values
            has_time = True
        else:
            display_data = data_array.values
            has_time = False

        # 创建图表框架
        main_frame = ttk.Frame(viewer_window)
        main_frame.pack(fill=BOTH, expand=True)

        # 左侧：地图
        map_frame = ttk.Frame(main_frame)
        map_frame.pack(side=LEFT, fill=BOTH, expand=True)

        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(display_data, cmap='viridis')
        ax.set_title("点击查看像元信息", fontsize=12, fontweight='bold')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.axis('off')

        # 右侧：信息面板
        info_frame = ttk.LabelFrame(main_frame, text="像元信息", padding=10)
        info_frame.pack(side=RIGHT, fill=Y, padx=5, pady=5)

        info_text = tk.Text(info_frame, width=35, height=30, font=("Consolas", 9))
        info_text.pack(fill=BOTH, expand=True)
        info_text.insert("1.0", "点击地图上的任意位置查看详细信息...")

        # 点击事件处理
        click_marker = None

        def on_click(event):
            nonlocal click_marker

            if event.inaxes == ax:
                col = int(event.xdata + 0.5)
                row = int(event.ydata + 0.5)

                ny, nx = display_data.shape
                if 0 <= row < ny and 0 <= col < nx:
                    # 清除之前的标记
                    if click_marker is not None:
                        click_marker.remove()

                    # 添加新标记
                    click_marker = ax.plot(col, row, 'r+', markersize=15,
                                           markeredgewidth=2)[0]
                    canvas.draw()

                    # 更新信息
                    info_text.delete("1.0", tk.END)

                    info_content = f"位置坐标\n{'=' * 30}\n"
                    info_content += f"行 (Y): {row}\n"
                    info_content += f"列 (X): {col}\n\n"

                    value = display_data[row, col]
                    info_content += f"当前值\n{'=' * 30}\n"
                    info_content += f"{value:.6f}\n\n"

                    # 如果有时间序列，显示时序统计
                    if has_time and data_stack is not None:
                        ts = data_stack[:, row, col].values
                        valid_ts = ts[~np.isnan(ts)]

                        if len(valid_ts) > 0:
                            info_content += f"时序统计\n{'=' * 30}\n"
                            info_content += f"均值: {np.mean(valid_ts):.6f}\n"
                            info_content += f"标准差: {np.std(valid_ts):.6f}\n"
                            info_content += f"最小值: {np.min(valid_ts):.6f}\n"
                            info_content += f"最大值: {np.max(valid_ts):.6f}\n"
                            info_content += f"有效期数: {len(valid_ts)}/{len(ts)}\n"

                    info_text.insert("1.0", info_content)

        fig.canvas.mpl_connect('button_press_event', on_click)

        # 嵌入matplotlib图表
        canvas = FigureCanvasTkAgg(fig, map_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=BOTH, expand=True, padx=5, pady=5)

        # 按钮
        btn_frame = ttk.Frame(viewer_window)
        btn_frame.pack(side=BOTTOM, fill=X, padx=5, pady=5)

        ttk.Button(btn_frame, text="关闭", command=viewer_window.destroy,
                   bootstyle=SECONDARY).pack(side=RIGHT, padx=5)

        return viewer_window


# ==================== 主应用程序类（增强版） ====================

class RemoteSensingAppEnhanced:
    """增强版遥感分析应用"""

    def __init__(self):
        self.root = tb.Window(
            title=f"时序遥感分析系统 V{Config.VERSION} - 增强版 @3S&ML",
            themename="cosmo",
            size=(1600, 950)
        )

        # 数据状态
        self.data_stack = None
        self.uploaded_files = []
        self.analysis_results = {}
        self.current_figures = []
        self.preprocessed_stack = None  # 预处理后的数据

        # 进度跟踪
        self.progress_tracker = ProgressTracker()
        self.progress_tracker.add_callback(self.update_progress_ui)

        # UI组件引用
        self.ui_components = {}

        # 参数存储
        self.current_parameters = {}

        self.setup_ui()

    def setup_ui(self):
        """设置UI"""
        self._create_menu_bar()
        self._create_header()
        self._create_main_layout()

    def _create_menu_bar(self):
        """创建菜单栏"""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        # 文件菜单
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="文件", menu=file_menu)
        file_menu.add_command(label="打开数据文件", command=self.select_files)
        file_menu.add_separator()
        file_menu.add_command(label="保存项目", command=self.save_project)
        file_menu.add_command(label="加载项目", command=self.load_project)
        file_menu.add_separator()
        file_menu.add_command(label="导出参数配置", command=self.export_parameters)
        file_menu.add_command(label="导入参数配置", command=self.import_parameters)
        file_menu.add_separator()
        file_menu.add_command(label="退出", command=self.root.quit)

        # 数据处理菜单
        process_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="数据处理", menu=process_menu)
        process_menu.add_command(label="数据平滑", command=self.open_smooth_dialog)
        process_menu.add_command(label="异常值检测", command=self.open_outlier_dialog)
        process_menu.add_command(label="数据插值", command=self.open_interpolation_dialog)
        process_menu.add_separator()
        process_menu.add_command(label="空间裁剪", command=self.open_spatial_subset_dialog)
        process_menu.add_command(label="时间筛选", command=self.open_temporal_subset_dialog)

        # 高级分析菜单
        advanced_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="高级分析", menu=advanced_menu)
        advanced_menu.add_command(label="时间序列聚类", command=self.open_clustering_dialog)
        advanced_menu.add_command(label="生成时序动画", command=self.open_animation_dialog)

        # 工具菜单
        tools_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="工具", menu=tools_menu)
        tools_menu.add_command(label="交互式地图查看", command=self.open_interactive_map)
        tools_menu.add_command(label="生成分析报告", command=self.generate_report)

        # 帮助菜单
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="帮助", menu=help_menu)
        help_menu.add_command(label="使用说明", command=self.show_help)
        help_menu.add_command(label="关于", command=self.show_about)

    def _create_header(self):
        """创建标题栏"""
        header_frame = ttk.Frame(self.root)
        header_frame.pack(fill=X, padx=10, pady=10)

        title_label = ttk.Label(
            header_frame,
            text=f"🛰️ 时序遥感分析系统 V{Config.VERSION} - 增强版",
            font=("Helvetica", 18, "bold")
        )
        title_label.pack()

        desc_label = ttk.Label(
            header_frame,
            text="Theil–Sen | Mann–Kendall | BFAST | FFT | STL | 聚类分析 | 动画生成",
            font=("Helvetica", 11)
        )
        desc_label.pack(pady=(5, 0))

    def _create_main_layout(self):
        """创建主布局"""
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=BOTH, expand=True, padx=10, pady=(0, 10))

        paned_window = ttk.PanedWindow(main_frame, orient=HORIZONTAL)
        paned_window.pack(fill=BOTH, expand=True)

        # 左侧控制面板
        left_frame = ttk.Frame(paned_window, width=340)
        paned_window.add(left_frame, weight=1)

        # 右侧结果显示面板
        right_frame = ttk.Frame(paned_window)
        paned_window.add(right_frame, weight=3)

        self._setup_left_panel(left_frame)
        self._setup_right_panel(right_frame)

    def _setup_left_panel(self, parent):
        """设置左侧控制面板"""
        # 创建可滚动区域
        canvas = tk.Canvas(parent, highlightthickness=0)
        scrollbar = ttk.Scrollbar(parent, orient=VERTICAL, command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side=LEFT, fill=BOTH, expand=True)
        scrollbar.pack(side=RIGHT, fill=Y)

        # 添加各个控制区域
        self._add_file_upload_section(scrollable_frame)
        self._add_data_info_section(scrollable_frame)
        self._add_analysis_control_section(scrollable_frame)
        self._add_pixel_analysis_section(scrollable_frame)

    def _add_file_upload_section(self, parent):
        """文件上传区域"""
        file_frame = ttk.LabelFrame(parent, text="📁 数据上传", padding=10)
        file_frame.pack(fill=X, pady=(0, 10), padx=5)

        ttk.Button(
            file_frame,
            text="选择 GeoTIFF 文件",
            command=self.select_files,
            bootstyle=PRIMARY,
            width=30
        ).pack(fill=X, pady=5)

        # 文件列表
        list_frame = ttk.Frame(file_frame)
        list_frame.pack(fill=X, pady=5)

        list_scroll = ttk.Scrollbar(list_frame)
        list_scroll.pack(side=RIGHT, fill=Y)

        self.file_listbox = tk.Listbox(
            list_frame,
            height=6,
            yscrollcommand=list_scroll.set
        )
        self.file_listbox.pack(side=LEFT, fill=X, expand=True)
        list_scroll.config(command=self.file_listbox.yview)

        # 按钮组
        btn_frame = ttk.Frame(file_frame)
        btn_frame.pack(fill=X, pady=5)

        ttk.Button(
            btn_frame,
            text="清除列表",
            command=self.clear_files,
            bootstyle=SECONDARY,
            width=14
        ).pack(side=LEFT, padx=2)

        ttk.Button(
            btn_frame,
            text="加载数据",
            command=self.load_data,
            bootstyle=SUCCESS,
            width=14
        ).pack(side=RIGHT, padx=2)

    def _add_data_info_section(self, parent):
        """数据信息区域"""
        info_frame = ttk.LabelFrame(parent, text="📊 数据信息", padding=10)
        info_frame.pack(fill=X, pady=(0, 10), padx=5)

        self.info_text = tk.Text(info_frame, height=6, wrap=tk.WORD,
                                 font=("Consolas", 9))
        self.info_text.pack(fill=X)
        self.info_text.insert("1.0", "请先上传数据文件...")
        self.info_text.config(state=tk.DISABLED)

    def _add_analysis_control_section(self, parent):
        """分析控制区域"""
        analysis_frame = ttk.LabelFrame(parent, text="🔧 分析控制", padding=10)
        analysis_frame.pack(fill=X, pady=(0, 10), padx=5)

        # 分析方法选择
        ttk.Label(analysis_frame, text="选择分析方法:",
                  font=("Helvetica", 10, "bold")).pack(anchor=W, pady=(0, 5))

        self.analysis_vars = {}
        analyses = [
            ("✓ Theil–Sen 趋势分析", "theilsen"),
            ("✓ Mann–Kendall 检验", "mk"),
            ("✓ BFAST 突变检测", "bfast"),
            ("✓ FFT 周期分析", "fft"),
            ("✓ STL 分解", "stl")
        ]

        for name, key in analyses:
            var = tk.BooleanVar(value=True)
            self.analysis_vars[key] = var
            cb = ttk.Checkbutton(analysis_frame, text=name, variable=var)
            cb.pack(anchor=W, pady=2)

        # STL参数设置
        ttk.Separator(analysis_frame, orient=HORIZONTAL).pack(fill=X, pady=10)

        param_frame = ttk.Frame(analysis_frame)
        param_frame.pack(fill=X, pady=5)

        ttk.Label(param_frame, text="STL周期:").pack(side=LEFT)
        self.stl_period_var = tk.IntVar(value=Config.STL_DEFAULT_PERIOD)
        period_spinbox = ttk.Spinbox(
            param_frame,
            from_=2,
            to=365,
            textvariable=self.stl_period_var,
            width=10
        )
        period_spinbox.pack(side=LEFT, padx=5)

        ttk.Separator(analysis_frame, orient=HORIZONTAL).pack(fill=X, pady=10)

        # 执行按钮
        btn_frame = ttk.Frame(analysis_frame)
        btn_frame.pack(fill=X, pady=5)

        self.run_btn = ttk.Button(
            btn_frame,
            text="🚀 执行分析",
            command=self.run_analysis,
            bootstyle=SUCCESS,
            width=14
        )
        self.run_btn.pack(side=LEFT, padx=2)

        self.cancel_btn = ttk.Button(
            btn_frame,
            text="⏹ 取消",
            command=self.cancel_analysis,
            bootstyle=DANGER,
            width=14,
            state=tk.DISABLED
        )
        self.cancel_btn.pack(side=RIGHT, padx=2)

        # 进度条
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(
            analysis_frame,
            variable=self.progress_var,
            maximum=100,
            bootstyle="success-striped"
        )
        self.progress_bar.pack(fill=X, pady=5)

        self.progress_label = ttk.Label(analysis_frame, text="",
                                        font=("Helvetica", 9))
        self.progress_label.pack()

    def _add_pixel_analysis_section(self, parent):
        """像元分析区域"""
        pixel_frame = ttk.LabelFrame(parent, text="🔎 像元级分析", padding=10)
        pixel_frame.pack(fill=X, padx=5)

        # 行坐标
        ttk.Label(pixel_frame, text="行坐标 (Y):",
                  font=("Helvetica", 9, "bold")).pack(anchor=W, pady=(0, 2))

        row_frame = ttk.Frame(pixel_frame)
        row_frame.pack(fill=X, pady=(0, 10))

        self.row_var = tk.IntVar(value=0)
        self.row_scale = ttk.Scale(
            row_frame,
            from_=0,
            to=100,
            variable=self.row_var,
            orient=HORIZONTAL
        )
        self.row_scale.pack(side=LEFT, fill=X, expand=True)

        self.row_label = ttk.Label(row_frame, text="0", width=5)
        self.row_label.pack(side=RIGHT, padx=5)

        self.row_var.trace_add("write", self._update_coord_labels)

        # 列坐标
        ttk.Label(pixel_frame, text="列坐标 (X):",
                  font=("Helvetica", 9, "bold")).pack(anchor=W, pady=(0, 2))

        col_frame = ttk.Frame(pixel_frame)
        col_frame.pack(fill=X, pady=(0, 10))

        self.col_var = tk.IntVar(value=0)
        self.col_scale = ttk.Scale(
            col_frame,
            from_=0,
            to=100,
            variable=self.col_var,
            orient=HORIZONTAL
        )
        self.col_scale.pack(side=LEFT, fill=X, expand=True)

        self.col_label = ttk.Label(col_frame, text="0", width=5)
        self.col_label.pack(side=RIGHT, padx=5)

        self.col_var.trace_add("write", self._update_coord_labels)

        # 分析按钮
        ttk.Button(
            pixel_frame,
            text="📈 分析选中像元",
            command=self.analyze_pixel,
            bootstyle=INFO,
            width=30
        ).pack(fill=X, pady=10)

    def _update_coord_labels(self, *args):
        """更新坐标标签"""
        self.row_label.config(text=str(self.row_var.get()))
        self.col_label.config(text=str(self.col_var.get()))

    def _setup_right_panel(self, parent):
        """设置右侧结果面板"""
        self.notebook = ttk.Notebook(parent)
        self.notebook.pack(fill=BOTH, expand=True)

        # 欢迎页（内容更丰富）
        self._create_welcome_tab()

        # 数据预览页
        self.preview_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.preview_frame, text="数据预览")

        # 结果标签页字典
        self.result_frames = {}

    def _create_welcome_tab(self):
        """创建欢迎标签页"""
        welcome_frame = ttk.Frame(self.notebook)
        self.notebook.add(welcome_frame, text="欢迎")

        text_widget = tk.Text(
            welcome_frame,
            wrap=tk.WORD,
            font=("Consolas", 10),
            padx=20,
            pady=20
        )
        text_widget.pack(fill=BOTH, expand=True)

        welcome_text = f"""
🎯 系统功能 V{Config.VERSION}

【核心分析】
• Theil–Sen趋势分析: 稳健的长期趋势估计
• Mann–Kendall检验: 趋势显著性检验  
• BFAST突变检测: 时间序列结构突变识别
• FFT周期分析: 周期性特征提取
• STL分解: 趋势-季节-残差分解

【新增功能】
• 数据预处理: Savitzky-Golay平滑、异常值检测、数据插值
• 时间序列聚类: K-means、层次聚类分析
• 动画生成: GIF/MP4时序动画导出
• 交互式地图: 像元点击查询
• 项目管理: 保存/加载完整分析项目
• 报告生成: 自动生成分析报告

📁 数据要求

• 格式: GeoTIFF (.tif, .tiff)
• 时间信息: 文件名包含可解析时间
  - 年度: NDVI_2000.tif, NDVI_2001.tif
  - 月度: NDVI_200001.tif, NDVI_2000_01.tif
  - 日期: NDVI_2000_001.tif (年_儒略日)
• 空间一致性: 相同范围和分辨率
• 建议: 预处理去除云和异常值

⚡ 快速开始

1. 菜单栏"文件" → "打开数据文件"
2. 选择时序GeoTIFF文件
3. 点击"加载数据"进行验证
4. 选择分析方法并设置参数
5. 点击"执行分析"开始计算
6. 查看结果并导出

🔧 高级功能

• 数据处理菜单: 平滑、异常值检测、插值
• 高级分析菜单: 聚类分析、动画生成
• 工具菜单: 交互式查看、报告生成
• 项目管理: 保存完整分析项目

💡 使用技巧

• 年度数据(≥10年): 适合趋势和突变分析
• 月度数据(≥24月): 适合所有分析
• 聚类前建议先平滑数据
• 可保存项目便于后续分析
• 支持批量导出所有结果

⚠️ 注意事项

• 大数据集需要较长计算时间
• 可随时点击"取消"中断分析
• 定期保存项目避免数据丢失
• 查看日志文件了解详细信息

📧 技术支持

Version: {Config.VERSION} | @3S&ML Team
日志文件: rs_analysis.log
        """

        text_widget.insert("1.0", welcome_text)
        text_widget.config(state=tk.DISABLED)

    # ==================== 核心功能方法（续） ====================

    def select_files(self):
        """选择文件"""
        files = filedialog.askopenfilenames(
            title="选择 GeoTIFF 文件",
            filetypes=[
                ("TIFF files", "*.tif *.tiff"),
                ("All files", "*.*")
            ]
        )

        if files:
            self.uploaded_files = list(files)
            self.update_file_list()

    def clear_files(self):
        """清除文件列表"""
        self.uploaded_files = []
        self.file_listbox.delete(0, tk.END)
        self.data_stack = None
        self.preprocessed_stack = None
        self.update_info_display("请先上传数据文件...")

    def update_file_list(self):
        """更新文件列表"""
        self.file_listbox.delete(0, tk.END)
        for file in self.uploaded_files:
            self.file_listbox.insert(tk.END, os.path.basename(file))

    def load_data(self):
        """加载数据"""
        if not self.uploaded_files:
            messagebox.showwarning("警告", "请先选择数据文件")
            return

        def load_thread():
            try:
                self.root.after(0, lambda: self.update_info_display("正在加载数据..."))

                # 提取时间并排序
                times = []
                valid_files = []

                for file in self.uploaded_files:
                    filename = os.path.basename(file)
                    time_val = TimeExtractor.extract_time(filename)
                    if time_val is not None:
                        times.append(time_val)
                        valid_files.append(file)
                    else:
                        logger.warning(f"无法提取时间: {filename}")

                if not valid_files:
                    self.root.after(0, lambda: messagebox.showerror(
                        "错误", "未检测到有效的时间信息"))
                    return

                sorted_indices = sorted(range(len(times)), key=lambda i: times[i])
                sorted_files = [valid_files[i] for i in sorted_indices]
                sorted_times = [times[i] for i in sorted_indices]

                # 读取数据
                data_list = []
                for i, file in enumerate(sorted_files):
                    try:
                        da = rxr.open_rasterio(
                            file,
                            chunks=Config.CHUNK_SIZE
                        ).squeeze()

                        if "band" in da.dims:
                            da = da.isel(band=0).drop_vars('band')

                        data_list.append(da)

                        progress = ((i + 1) / len(sorted_files)) * 100
                        self.root.after(0, lambda p=progress:
                        self.update_info_display(f"读取文件中... {p:.1f}%"))
                    except Exception as e:
                        logger.error(f"读取失败 {file}: {e}")

                if not data_list:
                    self.root.after(0, lambda: messagebox.showerror(
                        "错误", "没有成功读取任何文件"))
                    return

                # 堆叠数据
                stack = xr.concat(data_list, dim="time")
                stack = stack.assign_coords(time=sorted_times)
                stack = stack.transpose('time', 'y', 'x')

                self.data_stack = stack
                self.preprocessed_stack = None  # 重置预处理数据

                # 更新UI
                self.root.after(0, self.on_data_loaded)

            except Exception as e:
                logger.error(f"数据加载失败: {e}")
                self.root.after(0, lambda: messagebox.showerror(
                    "错误", f"数据加载失败:\n{str(e)}"))

        threading.Thread(target=load_thread, daemon=True).start()

    def on_data_loaded(self):
        """数据加载完成"""
        self.update_data_info()

        ny, nx = self.data_stack.sizes['y'], self.data_stack.sizes['x']
        self.row_scale.config(to=ny - 1)
        self.col_scale.config(to=nx - 1)
        self.row_var.set(ny // 2)
        self.col_var.set(nx // 2)

        self.show_data_preview()
        messagebox.showinfo("成功", "数据加载完成!")

    def update_data_info(self):
        """更新数据信息"""
        if self.data_stack is None:
            return

        times = self.data_stack.time.values
        data_frequency = self._detect_data_frequency(times)
        time_range = self._format_time_range(times)

        ny, nx = self.data_stack.sizes['y'], self.data_stack.sizes['x']
        n_time = self.data_stack.sizes['time']

        sample_data = self.data_stack.isel(time=0).values
        valid_pixels = np.sum(~np.isnan(sample_data))
        total_pixels = sample_data.size
        valid_percent = (valid_pixels / total_pixels) * 100

        info_text = f"""数据频率: {data_frequency}
时间序列: {n_time} 期
空间大小: {ny} × {nx} 像元
时间范围: {time_range}
有效像元: {valid_pixels:,} ({valid_percent:.1f}%)
数据类型: {self.data_stack.dtype}"""

        if self.preprocessed_stack is not None:
            info_text += "\n\n⚠️ 当前使用预处理数据"

        self.update_info_display(info_text)

    def _detect_data_frequency(self, times):
        """检测数据频率"""
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

    def _format_time_range(self, times):
        """格式化时间范围"""
        try:
            start = pd.to_datetime(str(times[0])).strftime('%Y-%m-%d')
            end = pd.to_datetime(str(times[-1])).strftime('%Y-%m-%d')
            return f"{start} 至 {end}"
        except:
            return f"{times[0]} 至 {times[-1]}"

    def update_info_display(self, text):
        """更新信息显示"""
        self.info_text.config(state=tk.NORMAL)
        self.info_text.delete("1.0", tk.END)
        self.info_text.insert("1.0", text)
        self.info_text.config(state=tk.DISABLED)

    def show_data_preview(self):
        """显示数据预览"""
        for widget in self.preview_frame.winfo_children():
            widget.destroy()

        try:
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle("数据预览", fontsize=14, fontweight='bold')

            # 第一期影像
            first_image = self.data_stack.isel(time=0)
            im1 = axes[0, 0].imshow(first_image.values, cmap='viridis')
            axes[0, 0].set_title("第一期影像")
            plt.colorbar(im1, ax=axes[0, 0], fraction=0.046)
            axes[0, 0].axis('off')

            # 最后一期影像
            last_image = self.data_stack.isel(time=-1)
            im2 = axes[0, 1].imshow(last_image.values, cmap='viridis')
            axes[0, 1].set_title("最后一期影像")
            plt.colorbar(im2, ax=axes[0, 1], fraction=0.046)
            axes[0, 1].axis('off')

            # 时序均值
            mean_image = self.data_stack.mean(dim='time')
            im3 = axes[1, 0].imshow(mean_image.values, cmap='viridis')
            axes[1, 0].set_title("时序均值")
            plt.colorbar(im3, ax=axes[1, 0], fraction=0.046)
            axes[1, 0].axis('off')

            # 随机像元时序
            ny, nx = self.data_stack.sizes['y'], self.data_stack.sizes['x']
            n_samples = min(5, ny * nx)

            for _ in range(n_samples):
                row = np.random.randint(0, ny)
                col = np.random.randint(0, nx)
                ts = self.data_stack[:, row, col].values

                if not np.all(np.isnan(ts)):
                    axes[1, 1].plot(ts, 'o-', markersize=3, alpha=0.7,
                                    label=f'({row}, {col})')

            axes[1, 1].set_title("随机像元时序")
            axes[1, 1].set_xlabel("时间索引")
            axes[1, 1].set_ylabel("值")
            axes[1, 1].legend(fontsize=8, loc='best')
            axes[1, 1].grid(True, alpha=0.3)

            plt.tight_layout()

            canvas = FigureCanvasTkAgg(fig, self.preview_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=BOTH, expand=True, padx=10, pady=10)

            self.current_figures.append(fig)

        except Exception as e:
            logger.error(f"数据预览失败: {e}")
            error_label = ttk.Label(
                self.preview_frame,
                text=f"预览生成失败:\n{str(e)}",
                font=("Helvetica", 10)
            )
            error_label.pack(expand=True)

        # main_enhanced_part2_continued.py - 增强版时序遥感分析系统（第二部分续）
        # 接续第二部分前半，完整的UI方法和主程序入口

        # ==================== 数据预处理对话框方法 ====================

    def open_smooth_dialog(self):
        """打开数据平滑对话框"""
        if self.data_stack is None:
            messagebox.showwarning("警告", "请先加载数据")
            return

        dialog = tb.Toplevel(self.root)
        dialog.title("数据平滑")
        dialog.geometry("400x350")
        dialog.transient(self.root)
        dialog.grab_set()

        # 方法选择
        ttk.Label(dialog, text="平滑方法:", font=("Helvetica", 10, "bold")).pack(pady=10)

        method_var = tk.StringVar(value="savgol")
        ttk.Radiobutton(dialog, text="Savitzky-Golay滤波",
                        variable=method_var, value="savgol").pack(anchor=W, padx=20)
        ttk.Radiobutton(dialog, text="移动平均",
                        variable=method_var, value="moving").pack(anchor=W, padx=20)

        # 参数设置
        param_frame = ttk.LabelFrame(dialog, text="参数设置", padding=10)
        param_frame.pack(fill=X, padx=20, pady=10)

        # Savitzky-Golay参数
        sg_frame = ttk.Frame(param_frame)
        sg_frame.pack(fill=X, pady=5)
        ttk.Label(sg_frame, text="窗口长度:").pack(side=LEFT)
        window_var = tk.IntVar(value=Config.SMOOTH_WINDOW)
        ttk.Spinbox(sg_frame, from_=3, to=51, increment=2,
                    textvariable=window_var, width=10).pack(side=LEFT, padx=5)

        ttk.Label(sg_frame, text="多项式阶数:").pack(side=LEFT, padx=(10, 0))
        poly_var = tk.IntVar(value=Config.SMOOTH_POLYORDER)
        ttk.Spinbox(sg_frame, from_=1, to=5,
                    textvariable=poly_var, width=10).pack(side=LEFT, padx=5)

        # 移动平均参数
        ma_frame = ttk.Frame(param_frame)
        ma_frame.pack(fill=X, pady=5)
        ttk.Label(ma_frame, text="窗口大小:").pack(side=LEFT)
        ma_window_var = tk.IntVar(value=3)
        ttk.Spinbox(ma_frame, from_=2, to=20,
                    textvariable=ma_window_var, width=10).pack(side=LEFT, padx=5)

        # 进度条
        progress_var = tk.DoubleVar()
        progress_bar = ttk.Progressbar(dialog, variable=progress_var, maximum=100)
        progress_bar.pack(fill=X, padx=20, pady=10)

        status_label = ttk.Label(dialog, text="")
        status_label.pack()

        def execute_smooth():
            method = method_var.get()

            def smooth_thread():
                try:
                    tracker = ProgressTracker()

                    def update_progress(name, pct):
                        dialog.after(0, lambda: progress_var.set(pct))
                        dialog.after(0, lambda: status_label.config(text=name))

                    tracker.add_callback(update_progress)

                    if method == "savgol":
                        result = DataPreprocessor.smooth_savgol(
                            self.data_stack,
                            window_var.get(),
                            poly_var.get(),
                            tracker
                        )
                    else:
                        result = DataPreprocessor.smooth_moving_average(
                            self.data_stack,
                            ma_window_var.get(),
                            tracker
                        )

                    self.preprocessed_stack = result

                    dialog.after(0, lambda: messagebox.showinfo(
                        "成功", "数据平滑完成！\n后续分析将使用平滑后的数据。"))
                    dialog.after(0, dialog.destroy)
                    dialog.after(0, self.update_data_info)

                except Exception as e:
                    dialog.after(0, lambda: messagebox.showerror(
                        "错误", f"平滑失败:\n{str(e)}"))

            threading.Thread(target=smooth_thread, daemon=True).start()

        # 按钮
        btn_frame = ttk.Frame(dialog)
        btn_frame.pack(pady=10)

        ttk.Button(btn_frame, text="执行", command=execute_smooth,
                   bootstyle=SUCCESS).pack(side=LEFT, padx=5)
        ttk.Button(btn_frame, text="取消", command=dialog.destroy,
                   bootstyle=SECONDARY).pack(side=LEFT, padx=5)

    def open_outlier_dialog(self):
        """打开异常值检测对话框"""
        if self.data_stack is None:
            messagebox.showwarning("警告", "请先加载数据")
            return

        dialog = tb.Toplevel(self.root)
        dialog.title("异常值检测与处理")
        dialog.geometry("450x400")
        dialog.transient(self.root)
        dialog.grab_set()

        # 检测方法
        ttk.Label(dialog, text="检测方法:", font=("Helvetica", 10, "bold")).pack(pady=10)

        method_var = tk.StringVar(value="zscore")
        ttk.Radiobutton(dialog, text="Z-Score方法",
                        variable=method_var, value="zscore").pack(anchor=W, padx=20)
        ttk.Radiobutton(dialog, text="IQR方法",
                        variable=method_var, value="iqr").pack(anchor=W, padx=20)

        # 阈值设置
        threshold_frame = ttk.Frame(dialog)
        threshold_frame.pack(fill=X, padx=20, pady=10)
        ttk.Label(threshold_frame, text="阈值:").pack(side=LEFT)
        threshold_var = tk.DoubleVar(value=Config.OUTLIER_THRESHOLD)
        ttk.Entry(threshold_frame, textvariable=threshold_var, width=10).pack(side=LEFT, padx=5)

        # 处理方法
        ttk.Label(dialog, text="处理方法:", font=("Helvetica", 10, "bold")).pack(pady=10)

        replace_var = tk.StringVar(value="interpolate")
        ttk.Radiobutton(dialog, text="插值替换",
                        variable=replace_var, value="interpolate").pack(anchor=W, padx=20)
        ttk.Radiobutton(dialog, text="均值替换",
                        variable=replace_var, value="mean").pack(anchor=W, padx=20)
        ttk.Radiobutton(dialog, text="设为NaN",
                        variable=replace_var, value="nan").pack(anchor=W, padx=20)

        # 进度条
        progress_var = tk.DoubleVar()
        progress_bar = ttk.Progressbar(dialog, variable=progress_var, maximum=100)
        progress_bar.pack(fill=X, padx=20, pady=10)

        status_label = ttk.Label(dialog, text="")
        status_label.pack()

        def execute_outlier_detection():
            def outlier_thread():
                try:
                    tracker = ProgressTracker()

                    def update_progress(name, pct):
                        dialog.after(0, lambda: progress_var.set(pct))
                        dialog.after(0, lambda: status_label.config(text=name))

                    tracker.add_callback(update_progress)

                    # 检测异常值
                    outlier_mask = DataPreprocessor.detect_outliers(
                        self.data_stack,
                        method_var.get(),
                        threshold_var.get(),
                        tracker
                    )

                    # 移除异常值
                    result = DataPreprocessor.remove_outliers(
                        self.data_stack,
                        outlier_mask,
                        replace_var.get()
                    )

                    self.preprocessed_stack = result

                    outlier_count = np.sum(outlier_mask)
                    dialog.after(0, lambda: messagebox.showinfo(
                        "成功", f"异常值处理完成！\n检测到 {outlier_count} 个异常值。"))
                    dialog.after(0, dialog.destroy)
                    dialog.after(0, self.update_data_info)

                except Exception as e:
                    dialog.after(0, lambda: messagebox.showerror(
                        "错误", f"异常值处理失败:\n{str(e)}"))

            threading.Thread(target=outlier_thread, daemon=True).start()

        # 按钮
        btn_frame = ttk.Frame(dialog)
        btn_frame.pack(pady=10)

        ttk.Button(btn_frame, text="执行", command=execute_outlier_detection,
                   bootstyle=SUCCESS).pack(side=LEFT, padx=5)
        ttk.Button(btn_frame, text="取消", command=dialog.destroy,
                   bootstyle=SECONDARY).pack(side=LEFT, padx=5)

    def open_interpolation_dialog(self):
        """打开数据插值对话框"""
        if self.data_stack is None:
            messagebox.showwarning("警告", "请先加载数据")
            return

        dialog = tb.Toplevel(self.root)
        dialog.title("数据插值")
        dialog.geometry("350x250")
        dialog.transient(self.root)
        dialog.grab_set()

        ttk.Label(dialog, text="插值方法:", font=("Helvetica", 10, "bold")).pack(pady=10)

        method_var = tk.StringVar(value="linear")
        ttk.Radiobutton(dialog, text="线性插值",
                        variable=method_var, value="linear").pack(anchor=W, padx=20)
        ttk.Radiobutton(dialog, text="三次样条插值",
                        variable=method_var, value="cubic").pack(anchor=W, padx=20)
        ttk.Radiobutton(dialog, text="最近邻插值",
                        variable=method_var, value="nearest").pack(anchor=W, padx=20)

        progress_var = tk.DoubleVar()
        progress_bar = ttk.Progressbar(dialog, variable=progress_var, maximum=100)
        progress_bar.pack(fill=X, padx=20, pady=10)

        status_label = ttk.Label(dialog, text="")
        status_label.pack()

        def execute_interpolation():
            def interp_thread():
                try:
                    tracker = ProgressTracker()

                    def update_progress(name, pct):
                        dialog.after(0, lambda: progress_var.set(pct))
                        dialog.after(0, lambda: status_label.config(text=name))

                    tracker.add_callback(update_progress)

                    result = DataPreprocessor.interpolate_gaps(
                        self.data_stack,
                        method_var.get(),
                        tracker
                    )

                    self.preprocessed_stack = result

                    dialog.after(0, lambda: messagebox.showinfo(
                        "成功", "数据插值完成！"))
                    dialog.after(0, dialog.destroy)
                    dialog.after(0, self.update_data_info)

                except Exception as e:
                    dialog.after(0, lambda: messagebox.showerror(
                        "错误", f"插值失败:\n{str(e)}"))

            threading.Thread(target=interp_thread, daemon=True).start()

        btn_frame = ttk.Frame(dialog)
        btn_frame.pack(pady=10)

        ttk.Button(btn_frame, text="执行", command=execute_interpolation,
                   bootstyle=SUCCESS).pack(side=LEFT, padx=5)
        ttk.Button(btn_frame, text="取消", command=dialog.destroy,
                   bootstyle=SECONDARY).pack(side=LEFT, padx=5)

        def open_spatial_subset_dialog(self):
            """打开空间裁剪对话框"""
            if self.data_stack is None:
                messagebox.showwarning("警告", "请先加载数据")
                return

            dialog = tb.Toplevel(self.root)
            dialog.title("空间裁剪")
            dialog.geometry("400x300")
            dialog.transient(self.root)
            dialog.grab_set()

            ny, nx = self.data_stack.sizes['y'], self.data_stack.sizes['x']

            ttk.Label(dialog, text="设置裁剪范围:", font=("Helvetica", 10, "bold")).pack(pady=10)

            param_frame = ttk.Frame(dialog)
            param_frame.pack(fill=X, padx=20, pady=10)

            ttk.Label(param_frame, text="起始行:").grid(row=0, column=0, sticky=W, pady=5)
            row_min_var = tk.IntVar(value=0)
            ttk.Entry(param_frame, textvariable=row_min_var, width=10).grid(row=0, column=1, padx=5)

            ttk.Label(param_frame, text="结束行:").grid(row=1, column=0, sticky=W, pady=5)
            row_max_var = tk.IntVar(value=ny)
            ttk.Entry(param_frame, textvariable=row_max_var, width=10).grid(row=1, column=1, padx=5)

            ttk.Label(param_frame, text="起始列:").grid(row=2, column=0, sticky=W, pady=5)
            col_min_var = tk.IntVar(value=0)
            ttk.Entry(param_frame, textvariable=col_min_var, width=10).grid(row=2, column=1, padx=5)

            ttk.Label(param_frame, text="结束列:").grid(row=3, column=0, sticky=W, pady=5)
            col_max_var = tk.IntVar(value=nx)
            ttk.Entry(param_frame, textvariable=col_max_var, width=10).grid(row=3, column=1, padx=5)

            info_label = ttk.Label(dialog, text=f"原始大小: {ny} × {nx}",
                                   font=("Consolas", 9))
            info_label.pack(pady=5)

            def execute_subset():
                try:
                    bbox = (row_min_var.get(), row_max_var.get(),
                            col_min_var.get(), col_max_var.get())

                    result = DataPreprocessor.spatial_subset(self.data_stack, bbox)

                    if result.sizes['y'] == 0 or result.sizes['x'] == 0:
                        messagebox.showerror("错误", "裁剪范围无效")
                        return

                    self.data_stack = result
                    self.preprocessed_stack = None

                    messagebox.showinfo("成功",
                                        f"空间裁剪完成！\n新大小: {result.sizes['y']} × {result.sizes['x']}")
                    dialog.destroy()
                    self.update_data_info()
                    self.show_data_preview()

                except Exception as e:
                    messagebox.showerror("错误", f"裁剪失败:\n{str(e)}")

            btn_frame = ttk.Frame(dialog)
            btn_frame.pack(pady=10)

            ttk.Button(btn_frame, text="执行", command=execute_subset,
                       bootstyle=SUCCESS).pack(side=LEFT, padx=5)
            ttk.Button(btn_frame, text="取消", command=dialog.destroy,
                       bootstyle=SECONDARY).pack(side=LEFT, padx=5)

        def open_temporal_subset_dialog(self):
            """打开时间筛选对话框"""
            if self.data_stack is None:
                messagebox.showwarning("警告", "请先加载数据")
                return

            dialog = tb.Toplevel(self.root)
            dialog.title("时间筛选")
            dialog.geometry("400x250")
            dialog.transient(self.root)
            dialog.grab_set()

            times = self.data_stack.time.values

            ttk.Label(dialog, text="选择时间范围:", font=("Helvetica", 10, "bold")).pack(pady=10)

            param_frame = ttk.Frame(dialog)
            param_frame.pack(fill=X, padx=20, pady=10)

            ttk.Label(param_frame, text="起始索引:").grid(row=0, column=0, sticky=W, pady=5)
            start_var = tk.IntVar(value=0)
            ttk.Spinbox(param_frame, from_=0, to=len(times) - 1,
                        textvariable=start_var, width=10).grid(row=0, column=1, padx=5)

            ttk.Label(param_frame, text="结束索引:").grid(row=1, column=0, sticky=W, pady=5)
            end_var = tk.IntVar(value=len(times) - 1)
            ttk.Spinbox(param_frame, from_=0, to=len(times) - 1,
                        textvariable=end_var, width=10).grid(row=1, column=1, padx=5)

            info_label = ttk.Label(dialog, text=f"原始时间长度: {len(times)}",
                                   font=("Consolas", 9))
            info_label.pack(pady=5)

            def execute_subset():
                try:
                    start_idx = start_var.get()
                    end_idx = end_var.get()

                    if start_idx >= end_idx:
                        messagebox.showerror("错误", "起始索引必须小于结束索引")
                        return

                    result = self.data_stack.isel(time=slice(start_idx, end_idx + 1))

                    self.data_stack = result
                    self.preprocessed_stack = None

                    messagebox.showinfo("成功",
                                        f"时间筛选完成！\n新时间长度: {len(result.time)}")
                    dialog.destroy()
                    self.update_data_info()
                    self.show_data_preview()

                except Exception as e:
                    messagebox.showerror("错误", f"筛选失败:\n{str(e)}")

            btn_frame = ttk.Frame(dialog)
            btn_frame.pack(pady=10)

            ttk.Button(btn_frame, text="执行", command=execute_subset,
                       bootstyle=SUCCESS).pack(side=LEFT, padx=5)
            ttk.Button(btn_frame, text="取消", command=dialog.destroy,
                       bootstyle=SECONDARY).pack(side=LEFT, padx=5)

        # ==================== 高级分析对话框 ====================

        def open_clustering_dialog(self):
            """打开聚类分析对话框"""
            if self.data_stack is None:
                messagebox.showwarning("警告", "请先加载数据")
                return

            dialog = tb.Toplevel(self.root)
            dialog.title("时间序列聚类分析")
            dialog.geometry("450x400")
            dialog.transient(self.root)
            dialog.grab_set()

            # 聚类方法
            ttk.Label(dialog, text="聚类方法:", font=("Helvetica", 10, "bold")).pack(pady=10)

            method_var = tk.StringVar(value="kmeans")
            ttk.Radiobutton(dialog, text="K-means聚类",
                            variable=method_var, value="kmeans").pack(anchor=W, padx=20)
            ttk.Radiobutton(dialog, text="层次聚类",
                            variable=method_var, value="hierarchical").pack(anchor=W, padx=20)

            # 参数设置
            param_frame = ttk.LabelFrame(dialog, text="参数设置", padding=10)
            param_frame.pack(fill=X, padx=20, pady=10)

            ttk.Label(param_frame, text="聚类数量:").pack(side=LEFT)
            n_clusters_var = tk.IntVar(value=Config.CLUSTER_DEFAULT)
            ttk.Spinbox(param_frame, from_=2, to=20,
                        textvariable=n_clusters_var, width=10).pack(side=LEFT, padx=5)

            # 层次聚类链接方法
            linkage_frame = ttk.Frame(dialog)
            linkage_frame.pack(fill=X, padx=20, pady=5)
            ttk.Label(linkage_frame, text="链接方法:").pack(side=LEFT)
            linkage_var = tk.StringVar(value="ward")
            ttk.Combobox(linkage_frame, textvariable=linkage_var,
                         values=["ward", "complete", "average", "single"],
                         width=10, state="readonly").pack(side=LEFT, padx=5)

            # 进度条
            progress_var = tk.DoubleVar()
            progress_bar = ttk.Progressbar(dialog, variable=progress_var, maximum=100)
            progress_bar.pack(fill=X, padx=20, pady=10)

            status_label = ttk.Label(dialog, text="")
            status_label.pack()

            def execute_clustering():
                def cluster_thread():
                    try:
                        tracker = ProgressTracker()

                        def update_progress(name, pct):
                            dialog.after(0, lambda: progress_var.set(pct))
                            dialog.after(0, lambda: status_label.config(text=name))

                        tracker.add_callback(update_progress)

                        # 使用预处理数据或原始数据
                        data = self.preprocessed_stack if self.preprocessed_stack is not None else self.data_stack

                        if method_var.get() == "kmeans":
                            cluster_map, centers, metrics = TimeSeriesClusterer.kmeans_clustering(
                                data,
                                n_clusters_var.get(),
                                progress_tracker=tracker
                            )

                            # 保存结果
                            self.analysis_results['clustering'] = {
                                'map': cluster_map,
                                'centers': centers,
                                'metrics': metrics,
                                'method': 'K-means'
                            }

                            metrics_str = f"轮廓系数: {metrics['silhouette']:.3f}\nCH指数: {metrics['calinski_harabasz']:.1f}"

                        else:
                            cluster_map = TimeSeriesClusterer.hierarchical_clustering(
                                data,
                                n_clusters_var.get(),
                                linkage_var.get(),
                                tracker
                            )

                            self.analysis_results['clustering'] = {
                                'map': cluster_map,
                                'method': 'Hierarchical'
                            }

                            metrics_str = "层次聚类完成"

                        dialog.after(0, lambda: messagebox.showinfo(
                            "成功", f"聚类分析完成！\n{metrics_str}"))
                        dialog.after(0, dialog.destroy)
                        dialog.after(0, self.show_clustering_results)

                    except Exception as e:
                        logger.error(f"Clustering failed: {e}")
                        dialog.after(0, lambda: messagebox.showerror(
                            "错误", f"聚类分析失败:\n{str(e)}"))

                threading.Thread(target=cluster_thread, daemon=True).start()

            btn_frame = ttk.Frame(dialog)
            btn_frame.pack(pady=10)

            ttk.Button(btn_frame, text="执行", command=execute_clustering,
                       bootstyle=SUCCESS).pack(side=LEFT, padx=5)
            ttk.Button(btn_frame, text="取消", command=dialog.destroy,
                       bootstyle=SECONDARY).pack(side=LEFT, padx=5)

        def open_animation_dialog(self):
            """打开动画生成对话框"""
            if self.data_stack is None:
                messagebox.showwarning("警告", "请先加载数据")
                return

            dialog = tb.Toplevel(self.root)
            dialog.title("生成时序动画")
            dialog.geometry("400x350")
            dialog.transient(self.root)
            dialog.grab_set()

            # 参数设置
            param_frame = ttk.LabelFrame(dialog, text="动画参数", padding=10)
            param_frame.pack(fill=X, padx=20, pady=10)

            ttk.Label(param_frame, text="帧率 (FPS):").grid(row=0, column=0, sticky=W, pady=5)
            fps_var = tk.IntVar(value=Config.ANIMATION_FPS)
            ttk.Spinbox(param_frame, from_=1, to=30,
                        textvariable=fps_var, width=10).grid(row=0, column=1, padx=5)

            ttk.Label(param_frame, text="分辨率 (DPI):").grid(row=1, column=0, sticky=W, pady=5)
            dpi_var = tk.IntVar(value=100)
            ttk.Spinbox(param_frame, from_=50, to=300, increment=50,
                        textvariable=dpi_var, width=10).grid(row=1, column=1, padx=5)

            ttk.Label(param_frame, text="颜色方案:").grid(row=2, column=0, sticky=W, pady=5)
            cmap_var = tk.StringVar(value="viridis")
            ttk.Combobox(param_frame, textvariable=cmap_var,
                         values=["viridis", "RdYlGn", "RdBu_r", "plasma", "coolwarm"],
                         width=10, state="readonly").grid(row=2, column=1, padx=5)

            ttk.Label(param_frame, text="输出格式:").grid(row=3, column=0, sticky=W, pady=5)
            format_var = tk.StringVar(value="gif")
            ttk.Combobox(param_frame, textvariable=format_var,
                         values=["gif", "mp4"],
                         width=10, state="readonly").grid(row=3, column=1, padx=5)

            # 进度条
            progress_var = tk.DoubleVar()
            progress_bar = ttk.Progressbar(dialog, variable=progress_var, maximum=100)
            progress_bar.pack(fill=X, padx=20, pady=10)

            status_label = ttk.Label(dialog, text="")
            status_label.pack()

            def execute_animation():
                # 选择保存位置
                file_ext = format_var.get()
                file_path = filedialog.asksaveasfilename(
                    defaultextension=f".{file_ext}",
                    filetypes=[(f"{file_ext.upper()} files", f"*.{file_ext}"),
                               ("All files", "*.*")],
                    initialfile=f"timeseries_animation.{file_ext}"
                )

                if not file_path:
                    return

                def anim_thread():
                    try:
                        def update_progress(name, pct):
                            dialog.after(0, lambda: progress_var.set(pct))
                            dialog.after(0, lambda: status_label.config(text=name))

                        data = self.preprocessed_stack if self.preprocessed_stack is not None else self.data_stack

                        result_path = AnimationGenerator.create_timeseries_animation(
                            data,
                            file_path,
                            fps=fps_var.get(),
                            cmap=cmap_var.get(),
                            dpi=dpi_var.get(),
                            progress_callback=update_progress
                        )

                        dialog.after(0, lambda: messagebox.showinfo(
                            "成功", f"动画已保存:\n{result_path}"))
                        dialog.after(0, dialog.destroy)

                    except Exception as e:
                        logger.error(f"Animation generation failed: {e}")
                        dialog.after(0, lambda: messagebox.showerror(
                            "错误", f"动画生成失败:\n{str(e)}"))

                threading.Thread(target=anim_thread, daemon=True).start()

            btn_frame = ttk.Frame(dialog)
            btn_frame.pack(pady=10)

            ttk.Button(btn_frame, text="生成", command=execute_animation,
                       bootstyle=SUCCESS).pack(side=LEFT, padx=5)
            ttk.Button(btn_frame, text="取消", command=dialog.destroy,
                       bootstyle=SECONDARY).pack(side=LEFT, padx=5)

        # ==================== 核心分析执行 ====================

        def run_analysis(self):
            """执行分析"""
            if self.data_stack is None:
                messagebox.showwarning("警告", "请先加载数据")
                return

            selected = [k for k, v in self.analysis_vars.items() if v.get()]
            if not selected:
                messagebox.showwarning("警告", "请选择至少一种分析方法")
                return

            self.progress_tracker.reset()
            self.progress_tracker.total_steps = len(selected)

            self.run_btn.config(state=tk.DISABLED)
            self.cancel_btn.config(state=tk.NORMAL)
            self.analysis_results.clear()

            def analysis_thread():
                try:
                    # 使用预处理数据或原始数据
                    data = self.preprocessed_stack if self.preprocessed_stack is not None else self.data_stack

                    step = 0

                    # Theil-Sen
                    if 'theilsen' in selected and not self.progress_tracker.is_cancelled:
                        self.progress_tracker.update("Theil-Sen趋势分析", step / len(selected))
                        slope, intercept = TrendAnalyzer.theil_sen(data, self.progress_tracker)
                        if not self.progress_tracker.is_cancelled:
                            self.analysis_results['theilsen'] = {
                                'slope': slope,
                                'intercept': intercept
                            }
                        step += 1

                    # Mann-Kendall
                    if 'mk' in selected and not self.progress_tracker.is_cancelled:
                        self.progress_tracker.update("Mann-Kendall检验", step / len(selected))
                        mk = TrendAnalyzer.mann_kendall(data, progress_tracker=self.progress_tracker)
                        if not self.progress_tracker.is_cancelled:
                            self.analysis_results['mk'] = mk
                        step += 1

                    # BFAST
                    if 'bfast' in selected and not self.progress_tracker.is_cancelled:
                        self.progress_tracker.update("BFAST突变检测", step / len(selected))
                        bfast = BreakpointDetector.bfast(data, progress_tracker=self.progress_tracker)
                        if not self.progress_tracker.is_cancelled:
                            self.analysis_results['bfast'] = bfast
                        step += 1

                    # FFT
                    if 'fft' in selected and not self.progress_tracker.is_cancelled:
                        self.progress_tracker.update("FFT周期分析", step / len(selected))
                        amp, period = FrequencyAnalyzer.fft(data, self.progress_tracker)
                        if not self.progress_tracker.is_cancelled:
                            self.analysis_results['fft'] = {
                                'amplitude': amp,
                                'period': period
                            }
                        step += 1

                    # STL
                    if 'stl' in selected and not self.progress_tracker.is_cancelled:
                        self.progress_tracker.update("STL分解", step / len(selected))
                        trend, seasonal, resid = STLDecomposer.decompose(
                            data,
                            self.stl_period_var.get(),
                            self.progress_tracker
                        )
                        if not self.progress_tracker.is_cancelled:
                            self.analysis_results['stl'] = {
                                'trend': trend,
                                'seasonal': seasonal,
                                'resid': resid
                            }
                        step += 1

                    # 完成
                    if not self.progress_tracker.is_cancelled:
                        self.progress_tracker.update("分析完成!", 1.0)
                        self.root.after(0, self.on_analysis_complete)
                    else:
                        self.root.after(0, self.on_analysis_cancelled)

                except Exception as e:
                    logger.error(f"Analysis failed: {e}")
                    self.root.after(0, lambda: messagebox.showerror(
                        "错误", f"分析过程出错:\n{str(e)}"))
                    self.root.after(0, self.reset_analysis_ui)

            threading.Thread(target=analysis_thread, daemon=True).start()

        def cancel_analysis(self):
            """取消分析"""
            self.progress_tracker.cancel()
            self.cancel_btn.config(state=tk.DISABLED)

        def on_analysis_complete(self):
            """分析完成"""
            self.reset_analysis_ui()
            self.show_analysis_results()
            messagebox.showinfo("完成", "所有分析已完成!")

        def on_analysis_cancelled(self):
            """分析取消"""
            self.reset_analysis_ui()
            self.analysis_results.clear()
            messagebox.showinfo("取消", "分析已取消")

        def reset_analysis_ui(self):
            """重置分析UI状态"""
            self.run_btn.config(state=tk.NORMAL)
            self.cancel_btn.config(state=tk.DISABLED)
            self.progress_var.set(0)
            self.progress_label.config(text="")

        def update_progress_ui(self, step_name, percentage):
            """更新进度UI"""
            self.progress_var.set(percentage)
            self.progress_label.config(text=step_name)

        def show_analysis_results(self):
            """显示分析结果（与原版类似，略）"""
            # 清除旧结果
            for frame in self.result_frames.values():
                try:
                    self.notebook.forget(frame)
                except:
                    pass
            self.result_frames.clear()

            # 创建新结果页
            for key, results in self.analysis_results.items():
                self._create_result_tab(key, results)

            if self.analysis_results:
                self._create_batch_download_tab()

        def show_clustering_results(self):
            """显示聚类结果"""
            if 'clustering' not in self.analysis_results:
                return

            results = self.analysis_results['clustering']
            cluster_map = results['map']

            # 创建聚类结果标签页
            frame = ttk.Frame(self.notebook)
            self.result_frames['clustering'] = frame
            self.notebook.add(frame, text="聚类分析")

            # 创建可视化
            if 'centers' in results:
                fig = Visualizer.create_cluster_visualization(
                    cluster_map.values,
                    results['centers'],
                    self.data_stack.time.values
                )
            else:
                fig = Visualizer.create_result_figure(
                    cluster_map,
                    "聚类结果",
                    cmap=Config.COLORMAPS['cluster']
                )

            canvas = FigureCanvasTkAgg(fig, frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=BOTH, expand=True, padx=10, pady=10)
            self.current_figures.append(fig)

            # 下载按钮
            btn_frame = ttk.Frame(frame)
            btn_frame.pack(pady=10)

            ttk.Button(
                btn_frame,
                text="📥 下载聚类地图",
                command=lambda: self.download_result(cluster_map, "cluster_map.tif"),
                bootstyle=PRIMARY
            ).pack(side=LEFT, padx=5)

            # 切换到聚类结果页
            self.notebook.select(frame)

        def _create_result_tab(self, key, results):
            """创建结果标签页（简化版）"""
            frame = ttk.Frame(self.notebook)
            self.result_frames[key] = frame
            self.notebook.add(frame, text=self._get_analysis_name(key))

            # 根据分析类型创建可视化
            if key == 'theilsen':
                fig = Visualizer.create_result_figure(results['slope'], "Theil-Sen斜率")
            elif key == 'mk':
                fig = Visualizer.create_result_figure(results, "Mann-Kendall检验", vmin=-1, vmax=1)
            elif key == 'bfast':
                fig = Visualizer.create_result_figure(results, "BFAST突变点")
            elif key == 'fft':
                fig = Visualizer.create_multi_panel_figure(
                    [results['amplitude'], results['period']],
                    ['FFT振幅', 'FFT周期'],
                    ['hot', 'cool']
                )
            elif key == 'stl':
                fig = Visualizer.create_multi_panel_figure(
                    [results['trend'], results['seasonal'], results['resid']],
                    ['趋势', '季节', '残差'],
                    ['RdYlBu', 'RdYlBu', 'RdYlBu']
                )

            canvas = FigureCanvasTkAgg(fig, frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=BOTH, expand=True, padx=10, pady=10)
            self.current_figures.append(fig)

        def _get_analysis_name(self, key):
            """获取分析名称"""
            names = {
                'theilsen': 'Theil-Sen',
                'mk': 'Mann-Kendall',
                'bfast': 'BFAST',
                'fft': 'FFT',
                'stl': 'STL'
            }
            return names.get(key, key)

        def _create_batch_download_tab(self):
            """创建批量下载标签页（简化版）"""
            batch_frame = ttk.Frame(self.notebook)
            self.notebook.add(batch_frame, text="📦 批量下载")

            ttk.Label(
                batch_frame,
                text="批量下载所有分析结果",
                font=("Helvetica", 14, "bold")
            ).pack(pady=20)

            ttk.Button(
                batch_frame,
                text="📥 下载全部结果 (ZIP)",
                command=self.batch_download_all,
                bootstyle=SUCCESS,
                width=30
            ).pack(pady=20)

        def batch_download_all(self):
            """批量下载所有结果"""
            if not self.analysis_results:
                messagebox.showwarning("警告", "没有可下载的结果")
                return

            file_path = filedialog.asksaveasfilename(
                defaultextension=".zip",
                filetypes=[("ZIP files", "*.zip")],
                initialfile=f"analysis_results_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
            )

            if not file_path:
                return

            def download_thread():
                try:
                    with zipfile.ZipFile(file_path, 'w', zipfile.ZIP_DEFLATED) as zf:
                        for key, results in self.analysis_results.items():
                            if isinstance(results, dict):
                                for sub_key, data in results.items():
                                    tif_data = DataExporter.to_geotiff(data, self.data_stack)
                                    zf.writestr(f"{key}_{sub_key}.tif", tif_data)
                            else:
                                tif_data = DataExporter.to_geotiff(results, self.data_stack)
                                zf.writestr(f"{key}.tif", tif_data)

                    self.root.after(0, lambda: messagebox.showinfo(
                        "成功", f"批量下载完成！\n{file_path}"))
                except Exception as e:
                    self.root.after(0, lambda: messagebox.showerror(
                        "错误", f"下载失败:\n{str(e)}"))

            threading.Thread(target=download_thread, daemon=True).start()

        def download_result(self, data_array, filename):
            """下载单个结果"""
            try:
                file_path = filedialog.asksaveasfilename(
                    defaultextension=".tif",
                    filetypes=[("TIFF files", "*.tif")],
                    initialfile=filename
                )

                if file_path:
                    tif_data = DataExporter.to_geotiff(data_array, self.data_stack)
                    with open(file_path, 'wb') as f:
                        f.write(tif_data)
                    messagebox.showinfo("成功", f"文件已保存:\n{file_path}")
            except Exception as e:
                messagebox.showerror("错误", f"下载失败:\n{str(e)}")

        def analyze_pixel(self):
            """分析像元"""
            if self.data_stack is None:
                messagebox.showwarning("警告", "请先加载数据")
                return

            row = self.row_var.get()
            col = self.col_var.get()

        def analysis_thread():
            try:
                data = self.preprocessed_stack if self.preprocessed_stack is not None else self.data_stack
                fig = Visualizer.create_pixel_analysis_figure(
                    data, row, col, self.stl_period_var.get()
                )
                self.root.after(0, lambda: self._show_pixel_window(fig, row, col))
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror(
                    "错误", f"像元分析失败:\n{str(e)}"))

        threading.Thread(target=analysis_thread, daemon=True).start()

    def _show_pixel_window(self, fig, row, col):
        """显示像元分析窗口"""
        win = tb.Toplevel(self.root)
        win.title(f"像元 ({int(row)}, {int(col)}) 分析")
        win.geometry("1100x850")

        canvas = FigureCanvasTkAgg(fig, win)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=BOTH, expand=True, padx=10, pady=10)

        btn_frame = ttk.Frame(win)
        btn_frame.pack(pady=10)

        ttk.Button(btn_frame, text="💾 保存图表",
                   command=lambda: self._save_figure(fig, f"pixel_{int(row)}_{int(col)}.png"),
                   bootstyle=PRIMARY).pack(side=LEFT, padx=5)
        ttk.Button(btn_frame, text="关闭",
                   command=win.destroy,
                   bootstyle=SECONDARY).pack(side=LEFT, padx=5)

        self.current_figures.append(fig)

    def _save_figure(self, fig, filename):
        """保存图表"""
        try:
            file_path = filedialog.asksaveasfilename(
                defaultextension=".png",
                filetypes=[("PNG files", "*.png"), ("PDF files", "*.pdf")],
                initialfile=filename
            )
            if file_path:
                fig.savefig(file_path, bbox_inches='tight', dpi=Config.DEFAULT_DPI)
                messagebox.showinfo("成功", f"图表已保存:\n{file_path}")
        except Exception as e:
            messagebox.showerror("错误", f"保存失败:\n{str(e)}")

    # ==================== 项目管理 ====================

    def save_project(self):
        """保存项目"""
        if self.data_stack is None:
            messagebox.showwarning("警告", "没有可保存的项目")
            return

        file_path = filedialog.asksaveasfilename(
            defaultextension=".rsproj",
            filetypes=[("RS Project", "*.rsproj"), ("All files", "*.*")],
            initialfile=f"project_{datetime.datetime.now().strftime('%Y%m%d')}.rsproj"
        )

        if file_path:
            try:
                parameters = {
                    'stl_period': self.stl_period_var.get(),
                    'selected_analyses': {k: v.get() for k, v in self.analysis_vars.items()}
                }

                ProjectManager.save_project(
                    self.data_stack,
                    self.analysis_results,
                    parameters,
                    file_path
                )

                messagebox.showinfo("成功", f"项目已保存:\n{file_path}")
            except Exception as e:
                messagebox.showerror("错误", f"保存失败:\n{str(e)}")

    def load_project(self):
        """加载项目"""
        file_path = filedialog.askopenfilename(
            filetypes=[("RS Project", "*.rsproj"), ("All files", "*.*")]
        )

        if file_path:
            try:
                project_data = ProjectManager.load_project(file_path)

                self.data_stack = project_data['data_stack']
                self.analysis_results = project_data['analysis_results']

                if 'parameters' in project_data:
                    params = project_data['parameters']
                    if 'stl_period' in params:
                        self.stl_period_var.set(params['stl_period'])

                self.update_data_info()
                self.show_data_preview()
                self.show_analysis_results()

                messagebox.showinfo("成功", "项目加载完成!")
            except Exception as e:
                messagebox.showerror("错误", f"加载失败:\n{str(e)}")

    def export_parameters(self):
        """导出参数"""
        file_path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json")],
            initialfile="parameters.json"
        )

        if file_path:
            try:
                parameters = {
                    'stl_period': self.stl_period_var.get(),
                    'selected_analyses': {k: v.get() for k, v in self.analysis_vars.items()}
                }
                ProjectManager.export_parameters(parameters, file_path)
                messagebox.showinfo("成功", f"参数已导出:\n{file_path}")
            except Exception as e:
                messagebox.showerror("错误", f"导出失败:\n{str(e)}")

    def import_parameters(self):
        """导入参数"""
        file_path = filedialog.askopenfilename(
            filetypes=[("JSON files", "*.json")]
        )

        if file_path:
            try:
                parameters = ProjectManager.import_parameters(file_path)
                if 'stl_period' in parameters:
                    self.stl_period_var.set(parameters['stl_period'])
                messagebox.showinfo("成功", "参数导入完成!")
            except Exception as e:
                messagebox.showerror("错误", f"导入失败:\n{str(e)}")

    # ==================== 工具方法 ====================

    def open_interactive_map(self):
        """打开交互式地图"""
        if self.data_stack is None:
            messagebox.showwarning("警告", "请先加载数据")
            return

        data = self.preprocessed_stack if self.preprocessed_stack is not None else self.data_stack
        InteractiveTools.create_interactive_map(data, self.root, data)

    def generate_report(self):
        """生成分析报告"""
        if not self.analysis_results:
            messagebox.showwarning("警告", "没有可用的分析结果")
            return

        file_path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt")],
            initialfile=f"report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        )

        if file_path:
            try:
                times = self.data_stack.time.values
                data_info = {
                    'time_range': self._format_time_range(times),
                    'n_time': len(times),
                    'ny': self.data_stack.sizes['y'],
                    'nx': self.data_stack.sizes['x']
                }

                ReportGenerator.generate_text_report(
                    self.analysis_results,
                    data_info,
                    file_path
                )

                messagebox.showinfo("成功", f"报告已生成:\n{file_path}")
            except Exception as e:
                messagebox.showerror("错误", f"报告生成失败:\n{str(e)}")

    def show_help(self):
        """显示帮助"""
        help_window = tb.Toplevel(self.root)
        help_window.title("使用说明")
        help_window.geometry("700x600")

        text = tk.Text(help_window, wrap=tk.WORD, font=("Consolas", 10), padx=20, pady=20)
        text.pack(fill=BOTH, expand=True)

        help_text = """
时序遥感分析系统 V3.0 - 使用说明

1. 数据上传
   - 点击"选择 GeoTIFF 文件"
   - 选择时序影像文件
   - 点击"加载数据"

2. 数据预处理（可选）
   - 菜单 → 数据处理 → 数据平滑/异常值检测/数据插值
   - 预处理后分析将使用处理后的数据

3. 执行分析
   - 选择分析方法
   - 设置参数（如STL周期）
   - 点击"执行分析"

4. 查看结果
   - 在右侧标签页查看各分析结果
   - 可下载单个或批量下载

5. 高级功能
   - 时间序列聚类：菜单 → 高级分析 → 时间序列聚类
   - 生成动画：菜单 → 高级分析 → 生成时序动画
   - 交互式查看：菜单 → 工具 → 交互式地图查看

6. 项目管理
   - 保存项目：菜单 → 文件 → 保存项目
   - 加载项目：菜单 → 文件 → 加载项目

快捷键：
  无

更多信息请查看软件文档或联系技术支持。
        """

        text.insert("1.0", help_text)
        text.config(state=tk.DISABLED)

    def show_about(self):
        """显示关于"""
        about_text = f"""
时序遥感分析系统 V{Config.VERSION}

开发团队: @3S&ML
版本: {Config.VERSION}
发布日期: 2024

主要功能:
• 时序趋势分析
• 突变检测
• 周期分析
• 数据预处理
• 聚类分析
• 动画生成

技术支持:
Email: support@example.com
        """
        messagebox.showinfo("关于", about_text)

    def run(self):
        """运行应用"""
        self.root.mainloop()

# ==================== 主程序入口 ====================

def main():
    """主函数"""
    try:
        logger.info(f"Starting Remote Sensing Analysis System V{Config.VERSION}")
        app = RemoteSensingAppEnhanced()
        app.run()
    except Exception as e:
        logger.error(f"Application startup failed: {e}")
        print(f"启动失败: {e}")

if __name__ == "__main__":
    main()
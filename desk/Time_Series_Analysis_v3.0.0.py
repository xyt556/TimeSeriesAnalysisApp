# main_optimized.py - 优化的时序遥感分析系统
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import ttkbootstrap as tb
# from PyQt5.QtGui.QIcon import themeName
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
from statsmodels.tsa.seasonal import STL
from tqdm import tqdm
import zipfile
from concurrent.futures import ThreadPoolExecutor
import logging

warnings.filterwarnings('ignore')

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'STHeiti', 'Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['figure.max_open_warning'] = 50


# ==================== 配置常量 ====================

class Config:
    """配置类"""
    MAX_WORKERS = 4  # 最大线程数
    CHUNK_SIZE = {'x': 512, 'y': 512}  # 数据块大小
    DEFAULT_DPI = 150  # 图像DPI
    NODATA_VALUE = -9999.0  # 空值
    MK_SIGNIFICANCE = 0.05  # Mann-Kendall显著性水平
    BFAST_THRESHOLD = 2.0  # BFAST阈值
    STL_DEFAULT_PERIOD = 12  # STL默认周期


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
            callback(step_name, percentage)

    def add_callback(self, callback):
        """添加回调函数"""
        self.callbacks.append(callback)

    def cancel(self):
        """取消操作"""
        self.is_cancelled = True

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


# ==================== 分析算法类 ====================

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
                    progress = (processed / total_pixels) * 100
                    progress_tracker.update("Theil-Sen分析中", progress)

        coords = {"y": stack.y, "x": stack.x}
        return (
            xr.DataArray(slope, dims=("y", "x"), coords=coords),
            xr.DataArray(intercept, dims=("y", "x"), coords=coords)
        )

    @staticmethod
    def mann_kendall(stack: xr.DataArray, significance=0.05, progress_tracker=None):
        """Mann-Kendall趋势检验"""
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
                    progress = (processed / total_pixels) * 100
                    progress_tracker.update("Mann-Kendall检验中", progress)

        return xr.DataArray(out, dims=("y", "x"), coords={"y": stack.y, "x": stack.x})


class BreakpointDetector:
    """突变点检测器"""

    @staticmethod
    def bfast(stack: xr.DataArray, threshold=2.0, progress_tracker=None):
        """BFAST突变检测"""
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
                    progress = (processed / total_pixels) * 100
                    progress_tracker.update("BFAST突变检测中", progress)

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
                    if val > 1e18:  # 时间戳格式
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
                    progress = (processed / total_pixels) * 100
                    progress_tracker.update("FFT分析中", progress)

        coords = {"y": stack.y, "x": stack.x}
        return (
            xr.DataArray(amp, dims=("y", "x"), coords=coords),
            xr.DataArray(period, dims=("y", "x"), coords=coords)
        )


class STLDecomposer:
    """STL分解器"""

    @staticmethod
    def decompose(stack: xr.DataArray, period=12, progress_tracker=None):
        """STL分解"""
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
                    progress = (processed / total_pixels) * 100
                    progress_tracker.update("STL分解中", progress)

        coords = {"y": stack.y, "x": stack.x}
        return (
            xr.DataArray(trend_mean, dims=("y", "x"), coords=coords),
            xr.DataArray(seasonal_mean, dims=("y", "x"), coords=coords),
            xr.DataArray(resid_std, dims=("y", "x"), coords=coords)
        )


# ==================== 数据导出类 ====================

class DataExporter:
    """数据导出器"""

    @staticmethod
    def to_geotiff(data_array, reference_stack=None, nodata=Config.NODATA_VALUE):
        """转换为GeoTIFF字节数据"""
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
            logger.error(f"GeoTIFF生成失败: {e}")
            return DataExporter._create_simple_tiff(arr2d, nodata)

    @staticmethod
    def _to_2d_array(da):
        """转换为2D数组"""
        if "time" in da.dims and "y" in da.dims and "x" in da.dims:
            return np.nanmean(da.values, axis=0)
        elif "y" in da.dims and "x" in da.dims:
            return da.values
        else:
            vals = da.values
            if vals.ndim >= 2:
                return np.nanmean(vals, axis=tuple(range(vals.ndim - 2)))
            return vals

    @staticmethod
    def _get_spatial_reference(data_array, reference_stack):
        """获取空间参考信息"""
        crs = None
        transform = None

        # 尝试从data_array获取
        if hasattr(data_array, 'rio') and data_array.rio.crs is not None:
            crs = data_array.rio.crs
            transform = data_array.rio.transform()

        # 尝试从reference_stack获取
        if crs is None and reference_stack is not None:
            try:
                ref_da = reference_stack.isel(time=0)
                if hasattr(ref_da, 'rio') and ref_da.rio.crs is not None:
                    crs = ref_da.rio.crs
                    transform = ref_da.rio.transform()
            except:
                pass

        # 从坐标推断
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
            logger.error(f"创建简单TIFF失败: {e}")
            return b''


# ==================== 可视化类 ====================

class Visualizer:
    """可视化器"""

    @staticmethod
    def create_result_figure(data_array, title, cmap='RdBu_r', vmin=None, vmax=None):
        """创建结果图表"""
        fig, ax = plt.subplots(figsize=(10, 8))

        data = Visualizer._prepare_data(data_array)

        im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=14, fontweight='bold')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.axis('off')

        plt.tight_layout()
        return fig

    @staticmethod
    def create_multi_panel_figure(data_arrays, titles, cmaps=None, figsize=(15, 5)):
        """创建多面板图表"""
        n_panels = len(data_arrays)
        fig, axes = plt.subplots(1, n_panels, figsize=figsize)

        if n_panels == 1:
            axes = [axes]

        if cmaps is None:
            cmaps = ['RdBu_r'] * n_panels

        for i, (data_array, title, cmap) in enumerate(zip(data_arrays, titles, cmaps)):
            data = Visualizer._prepare_data(data_array)
            im = axes[i].imshow(data, cmap=cmap)
            axes[i].set_title(title, fontsize=12, fontweight='bold')
            plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
            axes[i].axis('off')

        plt.tight_layout()
        return fig

    @staticmethod
    def create_pixel_analysis_figure(stack, row, col, period=12):
        """创建像元分析图表"""
        series = stack[:, row, col].values
        times = stack["time"].values
        time_labels = Visualizer._format_time_labels(times)

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'像元 ({int(row)}, {int(col)}) 时序分析',
                     fontsize=16, fontweight='bold')

        # 原始时序
        ax1.plot(time_labels, series, 'o-', linewidth=2, markersize=5,
                 color='#2E86AB', alpha=0.7)
        ax1.set_title("原始时序", fontsize=12, fontweight='bold')
        ax1.set_ylabel("值", fontsize=10)
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.tick_params(axis='x', rotation=45)

        # 趋势分析
        Visualizer._add_trend_plot(ax2, series, time_labels)

        # STL分解
        Visualizer._add_stl_plots(ax3, ax4, series, time_labels, period)

        plt.tight_layout()
        return fig

    @staticmethod
    def _prepare_data(data_array):
        """准备显示数据"""
        if isinstance(data_array, xr.DataArray):
            if "time" in data_array.dims:
                return np.nanmean(data_array.values, axis=0)
            return data_array.values
        return data_array

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
                        color='#2E86AB', label='原始数据')
                ax.plot(time_labels, trend_line, '--', linewidth=2,
                        color='#A23B72', label=f'趋势线 (斜率: {coeffs[0]:.4f})')
                ax.set_title("趋势分析", fontsize=12, fontweight='bold')
                ax.set_ylabel("值", fontsize=10)
                ax.legend(loc='best')
                ax.grid(True, alpha=0.3, linestyle='--')
                ax.tick_params(axis='x', rotation=45)

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
                ax3.set_title("STL - 趋势分量", fontsize=12, fontweight='bold')
                ax3.set_xlabel("时间", fontsize=10)
                ax3.set_ylabel("值", fontsize=10)
                ax3.grid(True, alpha=0.3, linestyle='--')
                ax3.tick_params(axis='x', rotation=45)
                ax3.legend()

                ax4.plot(time_labels, stl_result.seasonal, linewidth=2,
                         color='#C73E1D', label='季节分量')
                ax4.set_title("STL - 季节分量", fontsize=12, fontweight='bold')
                ax4.set_xlabel("时间", fontsize=10)
                ax4.set_ylabel("值", fontsize=10)
                ax4.grid(True, alpha=0.3, linestyle='--')
                ax4.tick_params(axis='x', rotation=45)
                ax4.legend()
        except Exception as e:
            error_msg = f"STL分析失败\n{str(e)}"
            ax3.text(0.5, 0.5, error_msg, ha='center', va='center',
                     transform=ax3.transAxes, fontsize=10)
            ax4.text(0.5, 0.5, error_msg, ha='center', va='center',
                     transform=ax4.transAxes, fontsize=10)


# ==================== 主应用程序类 ====================

class RemoteSensingApp:
    """遥感分析应用主类"""

    def __init__(self):
        self.root = tb.Window(
            title="时序遥感分析系统 V3.0 Pro @3S&ML",
            themename="cosmo"

        )
        # 移除size参数，添加最大化窗口
        self.root.state('zoomed')  # 在Windows系统上最大化窗口
        # 或者使用 self.root.attributes('-zoomed', True)  # 在某些系统上


        # 数据状态
        self.data_stack = None
        self.uploaded_files = []
        self.analysis_results = {}
        self.current_figures = []

        # 进度跟踪
        self.progress_tracker = ProgressTracker()
        self.progress_tracker.add_callback(self.update_progress_ui)

        # UI组件引用
        self.ui_components = {}

        self.setup_ui()

    def setup_ui(self):
        """设置UI"""
        self._create_header()
        self._create_main_layout()

    def _create_header(self):
        """创建标题栏"""
        header_frame = ttk.Frame(self.root)
        header_frame.pack(fill=X, padx=10, pady=10)

        title_label = ttk.Label(
            header_frame,
            text="🛰️ 时序遥感分析系统 V3.0 Pro",
            font=("Helvetica", 18, "bold")
        )
        title_label.pack()

        desc_label = ttk.Label(
            header_frame,
            text="Theil–Sen | Mann–Kendall | BFAST | FFT | STL",
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
        left_frame = ttk.Frame(paned_window, width=320)
        paned_window.add(left_frame, weight=1)

        # 右侧结果显示面板
        right_frame = ttk.Frame(paned_window)
        paned_window.add(right_frame, weight=3)

        self._setup_left_panel(left_frame)
        self._setup_right_panel(right_frame)

    def _setup_left_panel(self, parent):
        """设置左侧控制面板"""
        # 使用Canvas和Scrollbar创建可滚动区域
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
        """添加文件上传区域"""
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
        """添加数据信息区域"""
        info_frame = ttk.LabelFrame(parent, text="📊 数据信息", padding=10)
        info_frame.pack(fill=X, pady=(0, 10), padx=5)

        self.info_text = tk.Text(info_frame, height=6, wrap=tk.WORD,
                                 font=("Consolas", 9))
        self.info_text.pack(fill=X)
        self.info_text.insert("1.0", "请先上传数据文件...")
        self.info_text.config(state=tk.DISABLED)

    def _add_analysis_control_section(self, parent):
        """添加分析控制区域"""
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
        """添加像元分析区域"""
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

        # 欢迎页
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

        # 创建滚动文本框
        text_widget = tk.Text(
            welcome_frame,
            wrap=tk.WORD,
            font=("Consolas", 10),
            padx=20,
            pady=20
        )
        text_widget.pack(fill=BOTH, expand=True)

        welcome_text = """
🎯 系统功能

• Theil–Sen趋势分析: 计算稳健的长期趋势斜率
• Mann–Kendall检验: 检验趋势的统计显著性  
• BFAST突变检测: 检测时间序列中的结构突变点
• FFT周期分析: 分析数据的周期性特征
• STL分解: 分解为趋势、季节和残差分量

📁 数据要求

• 文件格式: GeoTIFF (.tif, .tiff)
• 时间信息: 文件名必须包含可解析的时间信息
  - 年度数据: NDVI_2000.tif, NDVI_2001.tif, ...
  - 月度数据: NDVI_200001.tif, NDVI_2000_01.tif, ...
  - 日期数据: NDVI_2000_001.tif (年_儒略日), ...
• 空间一致性: 所有文件必须具有相同的空间范围和分辨率
• 数据质量: 建议进行预处理(云掩膜、异常值去除等)

⚡ 使用流程

1. 点击"选择 GeoTIFF 文件"上传时序数据
2. 点击"加载数据"进行数据读取和验证
3. 系统自动检测数据频率和时间范围
4. 在左侧选择要运行的分析方法
5. 调整参数(如STL周期)
6. 点击"执行分析"开始计算
7. 在结果标签页查看分析结果
8. 下载单个结果或批量下载全部结果

💡 分析建议

• 年度数据(>= 10年): 适合趋势分析和突变检测
• 月度数据(>= 24个月): 适合所有分析,特别是STL和FFT
• STL周期设置: 月度数据用12,季度数据用4
• 像元分析: 可以查看单个像元的详细时序特征

⚠️ 注意事项

• 分析过程可能需要较长时间,请耐心等待
• 大数据集建议分块处理或降低分辨率
• 结果中的NaN值表示无效或无显著变化区域
• 可以随时点击"取消"按钮中断分析

📧 技术支持

如遇问题,请联系技术支持团队
Version: 3.0 Pro | @3S&ML Team
        """

        text_widget.insert("1.0", welcome_text)
        text_widget.config(state=tk.DISABLED)

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

                # 提取时间信息
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

                # 按时间排序
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

                # 更新UI
                self.root.after(0, self.on_data_loaded)

            except Exception as e:
                logger.error(f"数据加载失败: {e}")
                self.root.after(0, lambda: messagebox.showerror(
                    "错误", f"数据加载失败:\n{str(e)}"))

        threading.Thread(target=load_thread, daemon=True).start()

    def on_data_loaded(self):
        """数据加载完成后的处理"""
        # 更新数据信息
        self.update_data_info()

        # 更新像元分析滑块范围
        ny, nx = self.data_stack.sizes['y'], self.data_stack.sizes['x']
        self.row_scale.config(to=ny - 1)
        self.col_scale.config(to=nx - 1)
        self.row_var.set(ny // 2)
        self.col_var.set(nx // 2)

        # 显示数据预览
        self.show_data_preview()

        messagebox.showinfo("成功", "数据加载完成!")

    def update_data_info(self):
        """更新数据信息显示"""
        if self.data_stack is None:
            return

        times = self.data_stack.time.values

        # 判断数据频率
        data_frequency = self._detect_data_frequency(times)

        # 格式化时间范围
        time_range = self._format_time_range(times)

        # 统计信息
        ny, nx = self.data_stack.sizes['y'], self.data_stack.sizes['x']
        n_time = self.data_stack.sizes['time']

        # 计算数据统计
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
        # 清除之前的内容
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

            # 显示图表
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

    def run_analysis(self):
        """执行分析"""
        if self.data_stack is None:
            messagebox.showwarning("警告", "请先加载数据")
            return

        selected = [k for k, v in self.analysis_vars.items() if v.get()]
        if not selected:
            messagebox.showwarning("警告", "请选择至少一种分析方法")
            return

        # 重置进度跟踪器
        self.progress_tracker.reset()
        self.progress_tracker.total_steps = len(selected)

        # 更新UI状态
        self.run_btn.config(state=tk.DISABLED)
        self.cancel_btn.config(state=tk.NORMAL)
        self.analysis_results.clear()

        def analysis_thread():
            try:
                step = 0

                # Theil-Sen
                if 'theilsen' in selected and not self.progress_tracker.is_cancelled:
                    self.progress_tracker.update("Theil-Sen趋势分析", step / len(selected))
                    slope, intercept = TrendAnalyzer.theil_sen(
                        self.data_stack,
                        self.progress_tracker
                    )
                    if not self.progress_tracker.is_cancelled:
                        self.analysis_results['theilsen'] = {
                            'slope': slope,
                            'intercept': intercept
                        }
                    step += 1

                # Mann-Kendall
                if 'mk' in selected and not self.progress_tracker.is_cancelled:
                    self.progress_tracker.update("Mann-Kendall检验", step / len(selected))
                    mk = TrendAnalyzer.mann_kendall(
                        self.data_stack,
                        Config.MK_SIGNIFICANCE,
                        self.progress_tracker
                    )
                    if not self.progress_tracker.is_cancelled:
                        self.analysis_results['mk'] = mk
                    step += 1

                # BFAST
                if 'bfast' in selected and not self.progress_tracker.is_cancelled:
                    self.progress_tracker.update("BFAST突变检测", step / len(selected))
                    bfast = BreakpointDetector.bfast(
                        self.data_stack,
                        Config.BFAST_THRESHOLD,
                        self.progress_tracker
                    )
                    if not self.progress_tracker.is_cancelled:
                        self.analysis_results['bfast'] = bfast
                    step += 1

                # FFT
                if 'fft' in selected and not self.progress_tracker.is_cancelled:
                    self.progress_tracker.update("FFT周期分析", step / len(selected))
                    amp, period = FrequencyAnalyzer.fft(
                        self.data_stack,
                        self.progress_tracker
                    )
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
                        self.data_stack,
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
                logger.error(f"分析失败: {e}")
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
        """显示分析结果"""
        # 清除旧的结果标签页
        for frame in self.result_frames.values():
            try:
                self.notebook.forget(frame)
            except:
                pass
        self.result_frames.clear()

        # 创建新的结果标签页
        for key, results in self.analysis_results.items():
            self._create_result_tab(key, results)

        # 创建批量下载标签页
        if self.analysis_results:
            self._create_batch_download_tab()

    def _create_result_tab(self, analysis_key, results):
        """创建结果标签页"""
        frame = ttk.Frame(self.notebook)
        self.result_frames[analysis_key] = frame

        tab_name = self._get_analysis_name(analysis_key)
        self.notebook.add(frame, text=tab_name)

        # 创建滚动区域
        canvas = tk.Canvas(frame)
        scrollbar = ttk.Scrollbar(frame, orient=VERTICAL, command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side=LEFT, fill=BOTH, expand=True)
        scrollbar.pack(side=RIGHT, fill=Y)

        # 添加具体内容
        self._add_result_content(analysis_key, results, scrollable_frame)

    def _get_analysis_name(self, key):
        """获取分析名称"""
        names = {
            'theilsen': 'Theil–Sen',
            'mk': 'Mann–Kendall',
            'bfast': 'BFAST',
            'fft': 'FFT',
            'stl': 'STL'
        }
        return names.get(key, key)

    def _add_result_content(self, key, results, parent):
        """添加结果内容"""
        if key == 'theilsen':
            self._add_theilsen_results(results, parent)
        elif key == 'mk':
            self._add_mk_results(results, parent)
        elif key == 'bfast':
            self._add_bfast_results(results, parent)
        elif key == 'fft':
            self._add_fft_results(results, parent)
        elif key == 'stl':
            self._add_stl_results(results, parent)

    def _add_theilsen_results(self, results, parent):
        """添加Theil-Sen结果"""
        slope = results['slope']

        # 创建图表
        fig = Visualizer.create_result_figure(
            slope,
            "Theil–Sen 趋势斜率",
            cmap='RdBu_r'
        )

        canvas = FigureCanvasTkAgg(fig, parent)
        canvas.draw()
        canvas.get_tk_widget().pack(pady=10, padx=10)
        self.current_figures.append(fig)

        # 统计信息
        self._add_statistics(slope, parent, "斜率")

        # 下载按钮
        self._add_download_buttons(
            parent,
            [("斜率", slope, "theil_sen_slope.tif"),
             ("截距", results['intercept'], "theil_sen_intercept.tif")]
        )

    def _add_mk_results(self, results, parent):
        """添加Mann-Kendall结果"""
        mk_da = results

        fig = Visualizer.create_result_figure(
            mk_da,
            "Mann–Kendall 趋势检验",
            cmap='RdBu_r',
            vmin=-1,
            vmax=1
        )

        canvas = FigureCanvasTkAgg(fig, parent)
        canvas.draw()
        canvas.get_tk_widget().pack(pady=10, padx=10)
        self.current_figures.append(fig)

        # 趋势统计
        mk_values = mk_da.values
        valid_mask = ~np.isnan(mk_values)

        if np.any(valid_mask):
            valid = mk_values[valid_mask]
            stats_text = f"""
趋势统计:
  显著上升: {np.sum(valid == 1):,} 像元 ({np.sum(valid == 1) / len(valid) * 100:.1f}%)
  显著下降: {np.sum(valid == -1):,} 像元 ({np.sum(valid == -1) / len(valid) * 100:.1f}%)
  无显著趋势: {np.sum(valid == 0):,} 像元 ({np.sum(valid == 0) / len(valid) * 100:.1f}%)
            """

            stats_label = ttk.Label(
                parent,
                text=stats_text,
                font=("Consolas", 10),
                justify=LEFT
            )
            stats_label.pack(pady=10)

        self._add_download_buttons(
            parent,
            [("Mann-Kendall结果", mk_da, "mann_kendall.tif")]
        )

    def _add_bfast_results(self, results, parent):
        """添加BFAST结果"""
        break_da = results

        fig = Visualizer.create_result_figure(
            break_da,
            "BFAST 突变检测 - 突变年份",
            cmap='viridis'
        )

        canvas = FigureCanvasTkAgg(fig, parent)
        canvas.draw()
        canvas.get_tk_widget().pack(pady=10, padx=10)
        self.current_figures.append(fig)

        # 突变统计
        break_values = break_da.values
        valid_mask = ~np.isnan(break_values)

        if np.any(valid_mask):
            valid_years = break_values[valid_mask].astype(int)
            unique_years, counts = np.unique(valid_years, return_counts=True)

            stats_text = f"""
突变统计:
  检测到突变: {len(valid_years):,} 像元
  年份范围: {unique_years.min()} - {unique_years.max()}
            """

            stats_label = ttk.Label(
                parent,
                text=stats_text,
                font=("Consolas", 10),
                justify=LEFT
            )
            stats_label.pack(pady=10)

            # 年份分布图
            dist_fig, ax = plt.subplots(figsize=(12, 4))
            ax.bar(unique_years, counts, color='skyblue', edgecolor='navy', alpha=0.7)
            ax.set_xlabel('年份', fontsize=10)
            ax.set_ylabel('像元数量', fontsize=10)
            ax.set_title('突变年份分布', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            plt.xticks(rotation=45)
            plt.tight_layout()

            dist_canvas = FigureCanvasTkAgg(dist_fig, parent)
            dist_canvas.draw()
            dist_canvas.get_tk_widget().pack(pady=10, padx=10)
            self.current_figures.append(dist_fig)

        self._add_download_buttons(
            parent,
            [("BFAST突变点", break_da, "bfast_breakpoints.tif")]
        )

    def _add_fft_results(self, results, parent):
        """添加FFT结果"""
        amp = results['amplitude']
        period = results['period']

        fig = Visualizer.create_multi_panel_figure(
            [amp, period],
            ['FFT 振幅', 'FFT 主周期'],
            ['hot', 'cool'],
            figsize=(15, 6)
        )

        canvas = FigureCanvasTkAgg(fig, parent)
        canvas.draw()
        canvas.get_tk_widget().pack(pady=10, padx=10)
        self.current_figures.append(fig)

        self._add_download_buttons(
            parent,
            [("FFT振幅", amp, "fft_amplitude.tif"),
             ("FFT周期", period, "fft_period.tif")]
        )

    def _add_stl_results(self, results, parent):
        """添加STL结果"""
        trend = results['trend']
        seasonal = results['seasonal']
        resid = results['resid']

        fig = Visualizer.create_multi_panel_figure(
            [trend, seasonal, resid],
            ['趋势分量(均值)', '季节分量(均值)', '残差(标准差)'],
            ['RdYlBu', 'RdYlBu', 'RdYlBu'],
            figsize=(18, 5)
        )

        canvas = FigureCanvasTkAgg(fig, parent)
        canvas.draw()
        canvas.get_tk_widget().pack(pady=10, padx=10)
        self.current_figures.append(fig)

        self._add_download_buttons(
            parent,
            [("STL趋势", trend, "stl_trend.tif"),
             ("STL季节", seasonal, "stl_seasonal.tif"),
             ("STL残差", resid, "stl_resid.tif")]
        )

    def _add_statistics(self, data_array, parent, label):
        """添加统计信息"""
        values = data_array.values
        valid = values[~np.isnan(values)]

        if len(valid) > 0:
            stats_text = f"""
{label}统计:
  最小值: {np.min(valid):.6f}
  最大值: {np.max(valid):.6f}
  平均值: {np.mean(valid):.6f}
  标准差: {np.std(valid):.6f}
  有效像元: {len(valid):,}
            """

            stats_label = ttk.Label(
                parent,
                text=stats_text,
                font=("Consolas", 10),
                justify=LEFT
            )
            stats_label.pack(pady=10)

    def _add_download_buttons(self, parent, items):
        """添加下载按钮"""
        btn_frame = ttk.Frame(parent)
        btn_frame.pack(pady=10)

        for i, (name, data, filename) in enumerate(items):
            ttk.Button(
                btn_frame,
                text=f"📥 下载{name}",
                command=lambda d=data, f=filename: self.download_result(d, f),
                bootstyle=PRIMARY,
                width=20
            ).grid(row=i // 3, column=i % 3, padx=5, pady=5)

    def _create_batch_download_tab(self):
        """创建批量下载标签页"""
        batch_frame = ttk.Frame(self.notebook)
        self.notebook.add(batch_frame, text="📦 批量下载")

        # 标题
        ttk.Label(
            batch_frame,
            text="批量下载分析结果",
            font=("Helvetica", 14, "bold")
        ).pack(pady=20)

        # 选择框架
        select_frame = ttk.LabelFrame(
            batch_frame,
            text="选择要下载的结果",
            padding=20
        )
        select_frame.pack(fill=BOTH, expand=True, padx=20, pady=10)

        # 创建选择列表
        self.batch_vars = {}

        row = 0
        col = 0

        for key in self.analysis_results.keys():
            if key == 'theilsen':
                items = [("Theil-Sen斜率", "theilsen_slope"),
                         ("Theil-Sen截距", "theilsen_intercept")]
            elif key == 'mk':
                items = [("Mann-Kendall检验", "mk")]
            elif key == 'bfast':
                items = [("BFAST突变点", "bfast")]
            elif key == 'fft':
                items = [("FFT振幅", "fft_amp"),
                         ("FFT周期", "fft_period")]
            elif key == 'stl':
                items = [("STL趋势", "stl_trend"),
                         ("STL季节", "stl_seasonal"),
                         ("STL残差", "stl_resid")]
            else:
                continue

            for name, var_key in items:
                var = tk.BooleanVar(value=True)
                self.batch_vars[var_key] = var

                cb = ttk.Checkbutton(
                    select_frame,
                    text=name,
                    variable=var
                )
                cb.grid(row=row, column=col, sticky=W, padx=10, pady=5)

                col += 1
                if col >= 3:
                    col = 0
                    row += 1

        # 操作按钮
        btn_frame = ttk.Frame(batch_frame)
        btn_frame.pack(pady=20)

        ttk.Button(
            btn_frame,
            text="全选",
            command=lambda: self._select_all_batch(True),
            bootstyle=INFO,
            width=15
        ).pack(side=LEFT, padx=5)

        ttk.Button(
            btn_frame,
            text="全不选",
            command=lambda: self._select_all_batch(False),
            bootstyle=SECONDARY,
            width=15
        ).pack(side=LEFT, padx=5)

        ttk.Button(
            btn_frame,
            text="📥 下载为ZIP",
            command=self.batch_download,
            bootstyle=SUCCESS,
            width=15
        ).pack(side=LEFT, padx=5)

    def _select_all_batch(self, value):
        """全选/全不选批量下载项"""
        for var in self.batch_vars.values():
            var.set(value)

    def batch_download(self):
        """批量下载"""
        if not self.analysis_results:
            messagebox.showwarning("警告", "没有可下载的结果")
            return

        # 收集选中的结果
        selected = {}

        for key, var in self.batch_vars.items():
            if not var.get():
                continue

            if key == "theilsen_slope" and 'theilsen' in self.analysis_results:
                selected["theil_sen_slope"] = self.analysis_results['theilsen']['slope']
            elif key == "theilsen_intercept" and 'theilsen' in self.analysis_results:
                selected["theil_sen_intercept"] = self.analysis_results['theilsen']['intercept']
            elif key == "mk" and 'mk' in self.analysis_results:
                selected["mann_kendall"] = self.analysis_results['mk']
            elif key == "bfast" and 'bfast' in self.analysis_results:
                selected["bfast_breakpoints"] = self.analysis_results['bfast']
            elif key == "fft_amp" and 'fft' in self.analysis_results:
                selected["fft_amplitude"] = self.analysis_results['fft']['amplitude']
            elif key == "fft_period" and 'fft' in self.analysis_results:
                selected["fft_period"] = self.analysis_results['fft']['period']
            elif key == "stl_trend" and 'stl' in self.analysis_results:
                selected["stl_trend"] = self.analysis_results['stl']['trend']
            elif key == "stl_seasonal" and 'stl' in self.analysis_results:
                selected["stl_seasonal"] = self.analysis_results['stl']['seasonal']
            elif key == "stl_resid" and 'stl' in self.analysis_results:
                selected["stl_resid"] = self.analysis_results['stl']['resid']

        if not selected:
            messagebox.showwarning("警告", "请至少选择一个结果")
            return

        # 选择保存位置
        file_path = filedialog.asksaveasfilename(
            defaultextension=".zip",
            filetypes=[("ZIP files", "*.zip"), ("All files", "*.*")],
            initialfile=f"analysis_results_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
        )

        if not file_path:
            return

        # 在新线程中执行下载
        def download_thread():
            try:
                # 创建进度窗口
                progress_win = tb.Toplevel(self.root)
                progress_win.title("批量下载")
                progress_win.geometry("400x150")
                progress_win.transient(self.root)
                progress_win.grab_set()

                ttk.Label(
                    progress_win,
                    text="正在生成下载文件...",
                    font=("Helvetica", 11)
                ).pack(pady=20)

                prog_var = tk.DoubleVar()
                prog_bar = ttk.Progressbar(
                    progress_win,
                    variable=prog_var,
                    maximum=100
                )
                prog_bar.pack(fill=X, padx=20, pady=10)

                status_label = ttk.Label(progress_win, text="")
                status_label.pack()

                # 转换为TIFF
                tiff_data = {}
                total = len(selected)

                for i, (name, data_array) in enumerate(selected.items()):
                    prog_var.set((i / total) * 100)
                    status_label.config(text=f"处理: {name}")
                    progress_win.update()

                    tiff_bytes = DataExporter.to_geotiff(
                        data_array,
                        self.data_stack
                    )
                    tiff_data[f"{name}.tif"] = tiff_bytes

                # 创建ZIP
                status_label.config(text="创建ZIP文件...")
                progress_win.update()

                with zipfile.ZipFile(file_path, 'w', zipfile.ZIP_DEFLATED) as zf:
                    # README
                    readme = f"""时序遥感分析结果
生成时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
包含文件: {len(tiff_data)}

文件列表:
"""
                    for name in tiff_data.keys():
                        readme += f"  - {name}\n"

                    zf.writestr("README.txt", readme)

                    # 添加TIFF文件
                    for name, data in tiff_data.items():
                        zf.writestr(name, data)

                progress_win.destroy()
                messagebox.showinfo("成功", f"批量下载完成!\n保存位置: {file_path}")

            except Exception as e:
                logger.error(f"批量下载失败: {e}")
                if 'progress_win' in locals():
                    progress_win.destroy()
                self.root.after(0, lambda: messagebox.showerror(
                    "错误", f"批量下载失败:\n{str(e)}"))

        threading.Thread(target=download_thread, daemon=True).start()

    def download_result(self, data_array, filename):
        """下载单个结果"""
        try:
            file_path = filedialog.asksaveasfilename(
                defaultextension=".tif",
                filetypes=[("TIFF files", "*.tif"), ("All files", "*.*")],
                initialfile=filename
            )

            if file_path:
                tif_data = DataExporter.to_geotiff(data_array, self.data_stack)

                with open(file_path, 'wb') as f:
                    f.write(tif_data)

                messagebox.showinfo("成功", f"文件已保存:\n{file_path}")

        except Exception as e:
            logger.error(f"下载失败: {e}")
            messagebox.showerror("错误", f"下载失败:\n{str(e)}")

    def analyze_pixel(self):
        """分析选中像元"""
        if self.data_stack is None:
            messagebox.showwarning("警告", "请先加载数据")
            return

        row = self.row_var.get()
        col = self.col_var.get()

        def analysis_thread():
            try:
                fig = Visualizer.create_pixel_analysis_figure(
                    self.data_stack,
                    row,
                    col,
                    self.stl_period_var.get()
                )

                self.root.after(0, lambda: self._show_pixel_window(fig, row, col))

            except Exception as e:
                logger.error(f"像元分析失败: {e}")
                self.root.after(0, lambda: messagebox.showerror(
                    "错误", f"像元分析失败:\n{str(e)}"))

        threading.Thread(target=analysis_thread, daemon=True).start()

    def _show_pixel_window(self, fig, row, col):
        """显示像元分析窗口"""
        win = tb.Toplevel(self.root)
        win.title(f"像元 ({int(row)}, {int(col)}) 分析")
        win.geometry("1000x800")

        # 图表
        canvas = FigureCanvasTkAgg(fig, win)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=BOTH, expand=True, padx=10, pady=10)

        # 按钮
        btn_frame = ttk.Frame(win)
        btn_frame.pack(pady=10)

        ttk.Button(
            btn_frame,
            text="💾 保存图表",
            command=lambda: self._save_figure(fig, f"pixel_{int(row)}_{int(col)}.png"),
            bootstyle=PRIMARY
        ).pack(side=LEFT, padx=5)

        ttk.Button(
            btn_frame,
            text="关闭",
            command=win.destroy,
            bootstyle=SECONDARY
        ).pack(side=LEFT, padx=5)

        self.current_figures.append(fig)

    def _save_figure(self, fig, filename):
        """保存图表"""
        try:
            file_path = filedialog.asksaveasfilename(
                defaultextension=".png",
                filetypes=[
                    ("PNG files", "*.png"),
                    ("PDF files", "*.pdf"),
                    ("All files", "*.*")
                ],
                initialfile=filename
            )

            if file_path:
                fig.savefig(file_path, bbox_inches='tight', dpi=Config.DEFAULT_DPI)
                messagebox.showinfo("成功", f"图表已保存:\n{file_path}")

        except Exception as e:
            logger.error(f"保存失败: {e}")
            messagebox.showerror("错误", f"保存失败:\n{str(e)}")

    def run(self):
        """运行应用"""
        self.root.mainloop()


# ==================== 主程序入口 ====================

def main():
    """主函数"""
    try:
        app = RemoteSensingApp()
        app.run()
    except Exception as e:
        logger.error(f"应用启动失败: {e}")
        print(f"错误: {e}")


if __name__ == "__main__":
    main()
# main.py - 完整的时序遥感分析系统单机版
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

warnings.filterwarnings('ignore')

# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'STHeiti']
matplotlib.rcParams['axes.unicode_minus'] = False


# ==================== 分析工具函数 ====================

def theil_sen_trend(stack: xr.DataArray):
    """
    Theil-Sen趋势分析 - 保持空值掩码
    """
    data = stack.values
    time_idx = np.arange(data.shape[0])
    ny, nx = data.shape[1], data.shape[2]
    slope = np.full((ny, nx), np.nan, dtype=np.float32)
    intercept = np.full((ny, nx), np.nan, dtype=np.float32)

    # 创建空值掩码（在所有时间步都为空值的像元）
    nan_mask = np.all(np.isnan(data), axis=0)

    for i in range(ny):
        for j in range(nx):
            # 如果该像元在所有时间步都是空值，跳过
            if nan_mask[i, j]:
                continue

            ts = data[:, i, j]
            if np.isnan(ts).all():
                continue
            try:
                # 使用scipy的theilslopes
                res = stats.theilslopes(ts, time_idx)
                slope[i, j] = res[0]  # 斜率
                intercept[i, j] = res[1]  # 截距
            except Exception:
                continue

    coords = {"y": stack.y, "x": stack.x}
    slope_da = xr.DataArray(slope, dims=("y", "x"), coords=coords)
    intercept_da = xr.DataArray(intercept, dims=("y", "x"), coords=coords)
    return slope_da, intercept_da


def mann_kendall_test(stack: xr.DataArray):
    """
    Mann-Kendall趋势检验 - 确保返回正确的值范围
    """
    from scipy.stats import kendalltau
    data = stack.values
    ny, nx = data.shape[1], data.shape[2]
    out = np.full((ny, nx), np.nan, dtype=np.float32)  # 初始化为NaN
    time_idx = np.arange(data.shape[0])

    # 创建空值掩码
    nan_mask = np.all(np.isnan(data), axis=0)

    for i in range(ny):
        for j in range(nx):
            # 如果该像元在所有时间步都是空值，保持为NaN
            if nan_mask[i, j]:
                continue

            ts = data[:, i, j]
            if np.isnan(ts).all() or np.sum(~np.isnan(ts)) < 3:
                out[i, j] = np.nan  # 保持为NaN
                continue
            try:
                # 移除NaN值
                mask = ~np.isnan(ts)
                valid_ts = ts[mask]
                valid_time = time_idx[mask]

                tau, p_value = kendalltau(valid_time, valid_ts)

                if np.isnan(p_value) or np.isnan(tau):
                    out[i, j] = np.nan
                elif p_value < 0.05:  # 显著性水平0.05
                    out[i, j] = 1.0 if tau > 0 else -1.0
                else:
                    out[i, j] = 0.0
            except Exception:
                out[i, j] = np.nan

    return xr.DataArray(out, dims=("y", "x"),
                        coords={"y": stack.y, "x": stack.x})


def convert_times_to_years(times):
    """
    将各种时间格式转换为年份数组
    """
    years = []
    for t in times:
        if isinstance(t, np.datetime64):
            # 处理np.datetime64
            try:
                # 方法1: 直接提取年份
                year = t.astype('datetime64[Y]').astype(int) + 1970
                years.append(year)
            except:
                try:
                    # 方法2: 通过字符串转换
                    ts = pd.to_datetime(str(t))
                    years.append(ts.year)
                except:
                    years.append(2000)  # 默认值
        elif hasattr(t, 'year'):
            # 处理datetime对象
            years.append(t.year)
        else:
            # 处理数字或字符串
            try:
                years.append(int(t))
            except:
                try:
                    # 尝试解析字符串
                    ts = pd.to_datetime(str(t))
                    years.append(ts.year)
                except:
                    years.append(2000)  # 默认值

    return np.array(years)


def bfast_detection(stack: xr.DataArray, change_threshold=2.0):
    """
    BFAST突变检测 - 修复时间转换问题
    """
    # 获取时间坐标并转换为年份
    times = stack["time"].values
    years = convert_times_to_years(times)

    data = stack.values
    n_time = data.shape[0]
    ny, nx = data.shape[1], data.shape[2]
    break_data = np.full((ny, nx), np.nan, dtype=np.float32)

    # 创建空值掩码
    nan_mask = np.all(np.isnan(data), axis=0)

    for i in range(ny):
        for j in range(nx):
            # 如果该像元在所有时间步都是空值，跳过
            if nan_mask[i, j]:
                continue

            ts = data[:, i, j]
            if np.isnan(ts).all() or n_time < 4:
                continue

            try:
                if np.sum(~np.isnan(ts)) < 4:
                    continue

                # 基于残差的突变检测
                x = np.arange(n_time)
                mask = ~np.isnan(ts)
                if np.sum(mask) < 4:
                    continue

                # 线性拟合
                coeffs = np.polyfit(x[mask], ts[mask], 1)
                trend = np.polyval(coeffs, x)
                residuals = ts - trend

                # 检测残差的突变点
                residual_std = np.nanstd(residuals)
                if residual_std == 0:
                    continue

                # 寻找超过阈值的点
                z_scores = np.abs(residuals) / residual_std
                break_points = np.where(z_scores > change_threshold)[0]

                if len(break_points) > 0:
                    # 返回第一个显著突变点对应的年份（直接使用年份，不转换）
                    break_idx = break_points[0]
                    break_data[i, j] = float(years[break_idx])

            except Exception:
                continue

    return xr.DataArray(break_data, dims=("y", "x"),
                        coords={"y": stack.y, "x": stack.x})


def fix_bfast_results(break_da):
    """
    修复BFAST结果中的时间值
    """
    break_values = break_da.values
    break_values_fixed = np.full_like(break_values, np.nan)

    current_year = datetime.datetime.now().year

    for i in range(break_values.shape[0]):
        for j in range(break_values.shape[1]):
            val = break_values[i, j]
            if not np.isnan(val):
                # 处理各种可能的时间格式
                if val > 1000000000000000000:  # 可能是纳秒时间戳
                    try:
                        # 转换为datetime对象
                        dt = pd.to_datetime(val)
                        fixed_year = dt.year
                        # 检查年份是否合理
                        if 1900 <= fixed_year <= current_year + 1:
                            break_values_fixed[i, j] = fixed_year
                    except:
                        pass
                elif 1900 <= val <= current_year + 1:  # 已经是合理年份
                    break_values_fixed[i, j] = val
                # 其他情况保持NaN

    return xr.DataArray(break_values_fixed, dims=break_da.dims, coords=break_da.coords)


def fft_analysis(stack: xr.DataArray):
    """
    FFT周期分析 - 保持空值掩码
    """
    data = stack.values
    n = data.shape[0]
    ny, nx = data.shape[1], data.shape[2]
    amp = np.full((ny, nx), np.nan, dtype=np.float32)
    period = np.full((ny, nx), np.nan, dtype=np.float32)

    # 创建空值掩码
    nan_mask = np.all(np.isnan(data), axis=0)

    for i in range(ny):
        for j in range(nx):
            # 如果该像元在所有时间步都是空值，跳过
            if nan_mask[i, j]:
                continue

            ts = data[:, i, j]
            if np.isnan(ts).all():
                continue
            try:
                # 去趋势
                y = ts - np.nanmean(ts)
                yf = fftpack.fft(y)
                xf = fftpack.fftfreq(n, d=1)

                # 只取正频率
                half = n // 2
                power = np.abs(yf[:half])
                power[0] = 0  # 忽略直流分量

                if power.size <= 1:
                    continue

                # 找到主频率（忽略第一个频率）
                idx = np.argmax(power[1:]) + 1
                amp[i, j] = float(power[idx])

                freq = xf[idx]
                if freq != 0:
                    period[i, j] = float(1.0 / freq)
                else:
                    period[i, j] = np.nan

            except Exception:
                continue

    return xr.DataArray(amp, dims=("y", "x"), coords={"y": stack.y, "x": stack.x}), \
        xr.DataArray(period, dims=("y", "x"), coords={"y": stack.y, "x": stack.x})


def stl_decompose_pixelwise(stack: xr.DataArray, period=12):
    """
    STL分解 - 保持空值掩码
    """
    data = stack.values
    n, ny, nx = data.shape

    # 预分配结果数组 - 二维统计量
    trend_mean = np.full((ny, nx), np.nan, dtype=np.float32)
    seasonal_mean = np.full((ny, nx), np.nan, dtype=np.float32)
    resid_std = np.full((ny, nx), np.nan, dtype=np.float32)

    # 创建空值掩码
    nan_mask = np.all(np.isnan(data), axis=0)

    for i in range(ny):
        for j in range(nx):
            # 如果该像元在所有时间步都是空值，跳过
            if nan_mask[i, j]:
                continue

            ts = data[:, i, j]
            if np.isnan(ts).all() or np.sum(~np.isnan(ts)) < period * 2:
                continue
            try:
                # 填充缺失值用于STL
                ts_filled = ts.copy()
                mask = ~np.isnan(ts)
                if not np.all(mask):
                    x = np.arange(n)
                    ts_filled = np.interp(x, x[mask], ts[mask])

                stl = STL(ts_filled, period=period, robust=True)
                res = stl.fit()

                # 直接计算统计量
                trend_mean[i, j] = np.mean(res.trend)
                seasonal_mean[i, j] = np.mean(res.seasonal)
                resid_std[i, j] = np.std(res.resid)

            except Exception as e:
                continue

    coords = {"y": stack.y, "x": stack.x}
    trend_da = xr.DataArray(trend_mean, dims=("y", "x"), coords=coords)
    seasonal_da = xr.DataArray(seasonal_mean, dims=("y", "x"), coords=coords)
    resid_da = xr.DataArray(resid_std, dims=("y", "x"), coords=coords)

    return trend_da, seasonal_da, resid_da


# ==================== 可视化工具函数 ====================

def create_custom_cmap():
    """创建自定义颜色映射"""
    colors = ['#2166ac', '#4393c3', '#92c5de', '#d1e5f0', '#f7f7f7',
              '#fddbc7', '#f4a582', '#d6604d', '#b2182b']
    return matplotlib.colors.LinearSegmentedColormap.from_list('custom_rdbu', colors, N=256)


def _da_to_2d(da):
    """
    将xarray DataArray转换为2D numpy数组
    处理多维度情况
    """
    try:
        # 如果是三维数据，计算时间维度的均值
        if "time" in da.dims and "y" in da.dims and "x" in da.dims:
            return np.nanmean(da.values, axis=0)
        elif "y" in da.dims and "x" in da.dims:
            return da.values
        else:
            vals = da.values
            if vals.ndim >= 2:
                # 对多余维度取均值
                return np.nanmean(vals, axis=tuple(range(vals.ndim - 2)))
            return vals
    except Exception as e:
        print(f"数据转换错误: {e}")
        return np.array(da)


def create_simple_tif(arr2d, nodata=-9999.0):
    """创建简单的TIFF文件（无坐标系）"""
    try:
        profile = {
            'driver': 'GTiff',
            'dtype': rasterio.float32,
            'count': 1,
            'height': arr2d.shape[0],
            'width': arr2d.shape[1],
            'transform': from_origin(0, arr2d.shape[0], 1, 1),
            'crs': None,
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
        print(f"创建简单TIFF失败: {e}")
        return b''


def dataarray_to_bytes_tif(da, nodata=-9999.0):
    """
    完全重写的GeoTIFF生成函数
    确保保持原始坐标系和空间参考信息
    """
    arr2d = _da_to_2d(da)

    # 处理NaN值
    arr2d = np.where(np.isnan(arr2d), nodata, arr2d).astype(np.float32)

    try:
        # 从原始数据栈获取参考信息
        if 'data_stack' in globals() and data_stack is not None:
            ref_da = data_stack.isel(time=0)

            # 获取CRS和变换信息
            crs = None
            transform = None

            # 方法1: 使用rioxarray的属性
            if hasattr(ref_da, 'rio') and ref_da.rio.crs is not None:
                crs = ref_da.rio.crs
                transform = ref_da.rio.transform()

            # 方法2: 如果rioxarray不可用，尝试从坐标推断
            if crs is None and hasattr(ref_da, 'x') and hasattr(ref_da, 'y'):
                # 从坐标创建近似的变换
                if len(ref_da.x) > 1 and len(ref_da.y) > 1:
                    x_res = float(ref_da.x[1] - ref_da.x[0])
                    y_res = float(ref_da.y[0] - ref_da.y[1])  # 注意y方向
                    transform = from_origin(
                        float(ref_da.x[0]) - x_res / 2,
                        float(ref_da.y[0]) + y_res / 2,
                        x_res,
                        y_res
                    )

            # 创建profile
            profile = {
                'driver': 'GTiff',
                'dtype': rasterio.float32,
                'count': 1,
                'height': arr2d.shape[0],
                'width': arr2d.shape[1],
                'compress': 'lzw',
                'nodata': nodata
            }

            # 添加CRS和变换信息
            if crs is not None:
                profile['crs'] = crs
            if transform is not None:
                profile['transform'] = transform
            else:
                # 默认变换
                profile['transform'] = from_origin(0, arr2d.shape[0], 1, 1)

            # 写入内存文件
            memfile = MemoryFile()
            with memfile.open(**profile) as dst:
                dst.write(arr2d, 1)

            data = memfile.read()
            memfile.close()
            return data

        else:
            # 没有参考数据，创建默认TIFF
            return create_simple_tif(arr2d, nodata)

    except Exception as e:
        print(f"生成GeoTIFF失败: {e}")
        # 返回简单TIFF作为fallback
        return create_simple_tif(arr2d, nodata)


def fig_to_bytes_png(fig, dpi=150):
    """将matplotlib图形转换为PNG字节"""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=dpi,
                facecolor='white', edgecolor='none')
    buf.seek(0)
    data = buf.read()
    buf.close()
    return data


def create_download_zip(results_dict, filename="analysis_results.zip"):
    """
    创建包含所有分析结果的ZIP文件
    """
    import zipfile
    from datetime import datetime

    zip_buffer = io.BytesIO()

    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        # 添加时间戳文件
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        readme_content = f"""
遥感时序分析结果
生成时间: {timestamp}
包含的分析结果:
"""

        for key in results_dict.keys():
            readme_content += f"- {key}\n"

        zip_file.writestr("README.txt", readme_content)

        # 添加各分析结果
        for name, data in results_dict.items():
            if data is not None:
                zip_file.writestr(f"{name}.tif", data)

    zip_buffer.seek(0)
    return zip_buffer.getvalue()


# ==================== 主应用程序类 ====================

class RemoteSensingApp:
    def __init__(self):
        self.root = tb.Window(
            title="时序遥感分析系统_V2.0 @3S&ML",
            themename="cosmo",
            size=(1400, 900)
        )

        # 初始化状态
        self.analysis_results = {}
        self.data_stack = None
        self.uploaded_files = []
        self.current_figures = []

        self.setup_ui()

    def setup_ui(self):
        """设置用户界面"""
        # 创建主框架
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=BOTH, expand=True, padx=10, pady=10)

        # 标题
        title_label = ttk.Label(
            main_frame,
            text="🛰️ 时序遥感分析系统_V2.0 @3S&ML",
            font=("Helvetica", 16, "bold")
        )
        title_label.pack(pady=(0, 10))

        # 功能说明
        desc_label = ttk.Label(
            main_frame,
            text="功能模块：Theil–Sen趋势分析 | Mann–Kendall检验 | BFAST突变检测 | FFT周期分析 | STL分解",
            font=("Helvetica", 10)
        )
        desc_label.pack(pady=(0, 20))

        # 创建左右分栏
        paned_window = ttk.PanedWindow(main_frame, orient=HORIZONTAL)
        paned_window.pack(fill=BOTH, expand=True)

        # 左侧控制面板
        left_frame = ttk.Frame(paned_window, width=300)
        paned_window.add(left_frame, weight=1)

        # 右侧结果显示面板
        self.right_frame = ttk.Frame(paned_window)
        paned_window.add(self.right_frame, weight=3)

        self.setup_left_panel(left_frame)
        self.setup_right_panel(self.right_frame)

    def setup_left_panel(self, parent):
        """设置左侧控制面板"""
        # 文件上传区域
        file_frame = ttk.LabelFrame(parent, text="📁 数据上传", padding=10)
        file_frame.pack(fill=X, pady=(0, 10))

        ttk.Button(
            file_frame,
            text="选择 GeoTIFF 文件",
            command=self.select_files,
            bootstyle=PRIMARY
        ).pack(fill=X, pady=5)

        # 文件列表
        self.file_listbox = tk.Listbox(file_frame, height=8)
        self.file_listbox.pack(fill=X, pady=5)

        ttk.Button(
            file_frame,
            text="清除文件列表",
            command=self.clear_files,
            bootstyle=SECONDARY
        ).pack(fill=X)

        # 数据信息显示
        self.info_frame = ttk.LabelFrame(parent, text="📊 数据信息", padding=10)
        self.info_frame.pack(fill=X, pady=(0, 10))

        info_text = "请先上传数据文件"
        self.info_label = ttk.Label(self.info_frame, text=info_text, wraplength=280)
        self.info_label.pack(fill=X)

        # 分析控制区域
        analysis_frame = ttk.LabelFrame(parent, text="🔧 分析控制", padding=10)
        analysis_frame.pack(fill=X, pady=(0, 10))

        # 分析方法选择
        self.analysis_vars = {}
        analyses = [
            ("Theil–Sen 趋势分析", "theilsen"),
            ("Mann–Kendall 检验", "mk"),
            ("BFAST 突变检测", "bfast"),
            ("FFT 周期分析", "fft"),
            ("STL 分解", "stl")
        ]

        for name, key in analyses:
            var = tk.BooleanVar(value=True)
            self.analysis_vars[key] = var
            ttk.Checkbutton(
                analysis_frame,
                text=name,
                variable=var
            ).pack(anchor=W, pady=2)

        # STL周期参数
        self.stl_period_var = tk.IntVar(value=12)
        self.stl_frame = ttk.Frame(analysis_frame)
        self.stl_frame.pack(fill=X, pady=5)
        ttk.Label(self.stl_frame, text="STL 周期:").pack(side=LEFT)
        ttk.Entry(self.stl_frame, textvariable=self.stl_period_var, width=8).pack(side=LEFT, padx=5)

        # 执行分析按钮
        ttk.Button(
            analysis_frame,
            text="🚀 执行选中分析",
            command=self.run_analysis,
            bootstyle=SUCCESS
        ).pack(fill=X, pady=10)

        # 进度条
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(
            analysis_frame,
            variable=self.progress_var,
            maximum=100
        )
        self.progress_bar.pack(fill=X, pady=5)

        self.progress_label = ttk.Label(analysis_frame, text="")
        self.progress_label.pack()

        # 像元分析区域
        pixel_frame = ttk.LabelFrame(parent, text="🔎 像元级分析", padding=10)
        pixel_frame.pack(fill=X)

        ttk.Label(pixel_frame, text="行坐标 (Y):").pack(anchor=W)
        self.row_var = tk.IntVar(value=0)
        row_scale = ttk.Scale(
            pixel_frame,
            from_=0,
            to=100,
            variable=self.row_var,
            orient=HORIZONTAL,
            command=self.on_pixel_change
        )
        row_scale.pack(fill=X, pady=5)

        ttk.Label(pixel_frame, text="列坐标 (X):").pack(anchor=W)
        self.col_var = tk.IntVar(value=0)
        col_scale = ttk.Scale(
            pixel_frame,
            from_=0,
            to=100,
            variable=self.col_var,
            orient=HORIZONTAL,
            command=self.on_pixel_change
        )
        col_scale.pack(fill=X, pady=5)

        ttk.Button(
            pixel_frame,
            text="分析选中像元",
            command=self.analyze_pixel,
            bootstyle=INFO
        ).pack(fill=X, pady=10)

    def setup_right_panel(self, parent):
        """设置右侧结果显示面板"""
        # 创建笔记本控件用于标签页
        self.notebook = ttk.Notebook(parent)
        self.notebook.pack(fill=BOTH, expand=True)

        # 欢迎标签页
        welcome_frame = ttk.Frame(self.notebook)
        self.notebook.add(welcome_frame, text="欢迎")

        welcome_text = """
🎯 系统功能
• Theil–Sen趋势分析: 计算稳健的趋势斜率
• Mann–Kendall检验: 检验趋势显著性  
• BFAST突变检测: 检测时间序列中的突变点
• FFT周期分析: 分析周期性特征
• STL分解: 分解为趋势、季节和残差分量

📁 数据要求
• 文件格式: GeoTIFF (.tif, .tiff)
• 时间信息: 文件名必须包含时间信息
• 年度数据命名: NDVI_2000.tif, NDVI_2001.tif
• 月度数据命名: NDVI_200001.tif, NDVI_2000_01.tif
• 空间范围: 所有文件必须具有相同的空间范围和分辨率

⚡ 使用流程
1. 点击左侧"选择 GeoTIFF 文件"上传数据
2. 系统自动检测数据频率（年度/月度）
3. 选择要运行的分析方法
4. 点击"执行选中分析"
5. 查看结果并下载

💡 分析建议
• 年度数据: 适合趋势分析和突变检测
• 月度数据: 适合所有分析方法，特别是STL和FFT周期分析
        """

        welcome_label = ttk.Label(welcome_frame, text=welcome_text, justify=LEFT)
        welcome_label.pack(padx=20, pady=20, fill=BOTH, expand=True)

        # 数据预览标签页
        self.preview_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.preview_frame, text="数据预览")

        # 结果标签页（动态创建）
        self.result_frames = {}

    def extract_time(self, filename):
        """从文件名中提取时间信息"""
        # 与原始代码相同的实现
        m = re.search(r'(19\d{2}|20\d{2})_(\d{3})', filename)
        if m:
            year = int(m.group(1))
            day_of_year = int(m.group(2))
            try:
                date = datetime.datetime(year, 1, 1) + datetime.timedelta(days=day_of_year - 1)
                return date
            except:
                return datetime.datetime(year, 1, 1)

        m = re.search(r'(19\d{2}|20\d{2})_(\d{1,2})', filename)
        if m:
            year = int(m.group(1))
            month = int(m.group(2))
            return datetime.datetime(year, month, 1)

        m = re.search(r'(19\d{2}|20\d{2})(\d{2})', filename)
        if m:
            year = int(m.group(1))
            month = int(m.group(2))
            return datetime.datetime(year, month, 1)

        m = re.search(r'(19\d{2}|20\d{2})', filename)
        if m:
            year = int(m.group(0))
            return datetime.datetime(year, 1, 1)

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

    def select_files(self):
        """选择文件"""
        files = filedialog.askopenfilenames(
            title="选择 GeoTIFF 文件",
            filetypes=[("TIFF files", "*.tif *.tiff"), ("All files", "*.*")]
        )

        if files:
            self.uploaded_files = list(files)
            self.update_file_list()
            self.load_data()

    def clear_files(self):
        """清除文件列表"""
        self.uploaded_files = []
        self.file_listbox.delete(0, tk.END)
        self.data_stack = None
        self.info_label.config(text="请先上传数据文件")

    def update_file_list(self):
        """更新文件列表显示"""
        self.file_listbox.delete(0, tk.END)
        for file in self.uploaded_files:
            self.file_listbox.insert(tk.END, os.path.basename(file))

    def load_data(self):
        """加载数据"""
        if not self.uploaded_files:
            return

        def load_thread():
            try:
                # 提取时间信息
                times = []
                valid_files = []

                for file in self.uploaded_files:
                    filename = os.path.basename(file)
                    time_val = self.extract_time(filename)
                    if time_val is not None:
                        times.append(time_val)
                        valid_files.append(file)
                    else:
                        print(f"无法从文件名提取时间信息: {filename}")

                if not valid_files:
                    messagebox.showerror("错误", "未检测到有效的时间信息")
                    return

                # 按时间排序
                sorted_indices = sorted(range(len(times)), key=lambda i: times[i])
                sorted_files = [valid_files[i] for i in sorted_indices]
                sorted_times = [times[i] for i in sorted_indices]

                # 读取数据
                data_list = []
                for file in sorted_files:
                    try:
                        da = rxr.open_rasterio(file, chunks={'x': 512, 'y': 512}).squeeze()
                        if "band" in da.dims:
                            da = da.isel(band=0).drop_vars('band')
                        data_list.append(da)
                    except Exception as e:
                        print(f"读取文件失败 {file}: {e}")

                if not data_list:
                    messagebox.showerror("错误", "没有成功读取任何文件")
                    return

                # 堆叠数据
                stack = xr.concat(data_list, dim="time")
                stack = stack.assign_coords(time=sorted_times)
                stack = stack.transpose('time', 'y', 'x')

                self.data_stack = stack

                # 更新UI
                self.root.after(0, self.update_data_info)

            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("错误", f"数据加载失败: {e}"))

        threading.Thread(target=load_thread, daemon=True).start()

    def update_data_info(self):
        """更新数据信息显示"""
        if self.data_stack is None:
            return

        times = self.data_stack.time.values
        time_labels = []
        for t in times:
            if isinstance(t, np.datetime64):
                time_labels.append(np.datetime_as_string(t, unit='D'))
            else:
                time_labels.append(str(t))

        # 判断数据频率
        data_frequency = "年度数据"
        if len(times) > 1:
            if isinstance(times[0], np.datetime64):
                time_diff = times[1] - times[0]
                days_diff = time_diff / np.timedelta64(1, 'D')
                if 28 <= days_diff <= 31:
                    data_frequency = "月度数据"
                elif 360 <= days_diff <= 370:
                    data_frequency = "年度数据"

        info_text = f"""数据频率: {data_frequency}
时间序列长度: {self.data_stack.sizes['time']} 期
空间分辨率: {self.data_stack.sizes['y']} × {self.data_stack.sizes['x']}
时间范围: {time_labels[0]} 至 {time_labels[-1]}"""

        self.info_label.config(text=info_text)

        # 更新像元分析的滑块范围
        ny, nx = self.data_stack.sizes['y'], self.data_stack.sizes['x']
        for widget in self.root.winfo_children():
            if isinstance(widget, ttk.Scale):
                if "row" in str(widget):
                    widget.config(to=ny - 1)
                elif "col" in str(widget):
                    widget.config(to=nx - 1)

        # 显示数据预览
        self.show_data_preview()

    def show_data_preview(self):
        """显示数据预览"""
        # 清除之前的预览
        for widget in self.preview_frame.winfo_children():
            widget.destroy()

        # 创建图表
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # 空间分布
        first_image = self.data_stack.isel(time=0)
        im1 = ax1.imshow(first_image.values, cmap='viridis')
        ax1.set_title("第一期空间分布")
        plt.colorbar(im1, ax=ax1)
        ax1.axis('off')

        # 时间序列抽样
        ny, nx = self.data_stack.sizes['y'], self.data_stack.sizes['x']
        import random
        for i in range(3):
            row = random.randint(0, ny - 1)
            col = random.randint(0, nx - 1)
            ts = self.data_stack[:, row, col].values
            times = range(len(ts))
            ax2.plot(times, ts, 'o-', markersize=3, label=f'像元 ({row}, {col})')

        ax2.set_title("随机像元时间序列")
        ax2.set_xlabel("时间索引")
        ax2.set_ylabel("值")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        # 在TKinter中显示图表
        canvas = FigureCanvasTkAgg(fig, self.preview_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=BOTH, expand=True)

        self.current_figures.append(fig)

    def run_analysis(self):
        """执行分析"""
        if self.data_stack is None:
            messagebox.showwarning("警告", "请先上传数据文件")
            return

        selected_analyses = []
        for key, var in self.analysis_vars.items():
            if var.get():
                selected_analyses.append(key)

        if not selected_analyses:
            messagebox.showwarning("警告", "请选择至少一种分析方法")
            return

        # 在新线程中执行分析
        def analysis_thread():
            try:
                total_analyses = len(selected_analyses)
                current_progress = 0

                def update_progress(step_name, progress):
                    self.root.after(0, lambda: self.progress_var.set(progress * 100))
                    self.root.after(0, lambda: self.progress_label.config(text=step_name))

                # Theil–Sen 分析
                if 'theilsen' in selected_analyses:
                    update_progress("正在计算 Theil–Sen 趋势...", current_progress / total_analyses)
                    try:
                        slope_da, intercept_da = theil_sen_trend(self.data_stack)
                        self.analysis_results['theilsen'] = {
                            'slope': slope_da,
                            'intercept': intercept_da
                        }
                    except Exception as e:
                        print(f"Theil–Sen 分析失败: {e}")
                    current_progress += 1
                    update_progress("Theil–Sen 趋势分析完成", current_progress / total_analyses)

                # Mann–Kendall 分析
                if 'mk' in selected_analyses:
                    update_progress("正在计算 Mann–Kendall 检验...", current_progress / total_analyses)
                    try:
                        mk_da = mann_kendall_test(self.data_stack)
                        self.analysis_results['mk'] = mk_da
                    except Exception as e:
                        print(f"Mann–Kendall 检验失败: {e}")
                    current_progress += 1
                    update_progress("Mann–Kendall 检验完成", current_progress / total_analyses)

                # BFAST 分析
                if 'bfast' in selected_analyses:
                    update_progress("正在检测突变点...", current_progress / total_analyses)
                    try:
                        break_da = bfast_detection(self.data_stack)
                        break_da_fixed = fix_bfast_results(break_da)
                        self.analysis_results['bfast'] = break_da_fixed
                    except Exception as e:
                        print(f"BFAST 突变检测失败: {e}")
                    current_progress += 1
                    update_progress("BFAST 突变检测完成", current_progress / total_analyses)

                # FFT 分析
                if 'fft' in selected_analyses:
                    update_progress("正在进行 FFT 周期分析...", current_progress / total_analyses)
                    try:
                        amp_da, period_da = fft_analysis(self.data_stack)
                        self.analysis_results['fft'] = {
                            'amplitude': amp_da,
                            'period': period_da
                        }
                    except Exception as e:
                        print(f"FFT 周期分析失败: {e}")
                    current_progress += 1
                    update_progress("FFT 周期分析完成", current_progress / total_analyses)

                # STL 分解
                if 'stl' in selected_analyses:
                    update_progress("正在执行 STL 分解...", current_progress / total_analyses)
                    try:
                        trend_da, seasonal_da, resid_da = stl_decompose_pixelwise(
                            self.data_stack,
                            period=self.stl_period_var.get()
                        )
                        self.analysis_results['stl'] = {
                            'trend': trend_da,
                            'seasonal': seasonal_da,
                            'resid': resid_da
                        }
                    except Exception as e:
                        print(f"STL 分解失败: {e}")
                    current_progress += 1
                    update_progress("STL 分解完成", current_progress / total_analyses)

                # 完成
                update_progress("所有分析完成!", 1.0)
                self.root.after(0, self.show_analysis_results)

            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("错误", f"分析过程中出错: {e}"))

        threading.Thread(target=analysis_thread, daemon=True).start()

    def show_analysis_results(self):
        """显示分析结果"""
        # 清除之前的结果标签页
        for key in self.result_frames:
            if key in self.notebook.tabs():
                self.notebook.forget(key)
        self.result_frames.clear()

        # 为每种分析创建结果标签页
        for analysis_key, results in self.analysis_results.items():
            frame = ttk.Frame(self.notebook)
            self.result_frames[analysis_key] = frame
            self.notebook.add(frame, text=self.get_analysis_name(analysis_key))

            # 创建滚动框架
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

            # 添加结果内容
            self.add_analysis_content(analysis_key, results, scrollable_frame)

    def get_analysis_name(self, key):
        """获取分析方法的显示名称"""
        names = {
            'theilsen': 'Theil–Sen 趋势',
            'mk': 'Mann–Kendall 检验',
            'bfast': 'BFAST 突变检测',
            'fft': 'FFT 周期分析',
            'stl': 'STL 分解'
        }
        return names.get(key, key)

    def add_analysis_content(self, analysis_key, results, parent):
        """为特定分析添加内容"""
        if analysis_key == 'theilsen':
            self.add_theilsen_content(results, parent)
        elif analysis_key == 'mk':
            self.add_mk_content(results, parent)
        elif analysis_key == 'bfast':
            self.add_bfast_content(results, parent)
        elif analysis_key == 'fft':
            self.add_fft_content(results, parent)
        elif analysis_key == 'stl':
            self.add_stl_content(results, parent)

    def add_theilsen_content(self, results, parent):
        """添加Theil-Sen分析结果"""
        slope_da = results['slope']

        # 创建图表
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(slope_da.values, cmap='RdBu_r')
        ax.set_title("Theil–Sen 斜率")
        plt.colorbar(im, ax=ax)
        ax.axis('off')

        # 显示图表
        canvas = FigureCanvasTkAgg(fig, parent)
        canvas.draw()
        canvas.get_tk_widget().pack(pady=10)
        self.current_figures.append(fig)

        # 下载按钮
        ttk.Button(
            parent,
            text="下载斜率结果 (GeoTIFF)",
            command=lambda: self.download_result(slope_da, "theil_sen_slope.tif"),
            bootstyle=PRIMARY
        ).pack(pady=5)

    def add_mk_content(self, results, parent):
        """添加Mann-Kendall分析结果"""
        mk_da = results

        # 创建图表
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(mk_da.values, cmap='RdBu_r', vmin=-1, vmax=1)
        ax.set_title("Mann–Kendall 趋势 (1=上升, -1=下降, 0=不显著)")
        plt.colorbar(im, ax=ax)
        ax.axis('off')

        # 显示图表
        canvas = FigureCanvasTkAgg(fig, parent)
        canvas.draw()
        canvas.get_tk_widget().pack(pady=10)
        self.current_figures.append(fig)

        # 统计信息
        mk_values = mk_da.values
        valid_mask = ~np.isnan(mk_values)
        if np.any(valid_mask):
            valid_values = mk_values[valid_mask]
            stats_text = f"""趋势统计:
    显著上升: {np.sum(valid_values == 1)} 像元
    显著下降: {np.sum(valid_values == -1)} 像元
    无显著趋势: {np.sum(valid_values == 0)} 像元"""

            stats_label = ttk.Label(parent, text=stats_text, justify=LEFT)
            stats_label.pack(pady=5)

        # 下载按钮
        ttk.Button(
            parent,
            text="下载 Mann-Kendall 检验结果 (GeoTIFF)",
            command=lambda: self.download_result(mk_da, "mann_kendall_test.tif"),
            bootstyle=PRIMARY
        ).pack(pady=5)

    def add_bfast_content(self, results, parent):
        """添加BFAST分析结果"""
        break_da = results

        # 创建图表
        fig, ax = plt.subplots(figsize=(10, 8))

        # 处理BFAST结果数据
        break_values = break_da.values

        # 过滤异常值，确保显示合理的年份范围
        current_year = datetime.datetime.now().year
        display_values = np.where(
            (break_values >= 1900) & (break_values <= current_year + 1),
            break_values,
            np.nan
        )

        im = ax.imshow(display_values, cmap='viridis')
        ax.set_title("BFAST突变检测 - 突变年份 (NaN=无突变)")
        plt.colorbar(im, ax=ax)
        ax.axis('off')

        # 显示图表
        canvas = FigureCanvasTkAgg(fig, parent)
        canvas.draw()
        canvas.get_tk_widget().pack(pady=10)
        self.current_figures.append(fig)

        # 统计信息
        valid_mask = ~np.isnan(display_values)
        if np.any(valid_mask):
            valid_years = display_values[valid_mask]
            stats_text = f"""突变统计:
    检测到突变的像元: {len(valid_years)} 个
    突变年份范围: {int(np.nanmin(valid_years))} - {int(np.nanmax(valid_years))}年

    突变年份分布:"""

            # 计算年份分布
            unique_years, counts = np.unique(valid_years.astype(int), return_counts=True)
            for year, count in zip(unique_years, counts):
                stats_text += f"\n{year}年: {count} 像元"

            # 添加年份分布图表
            dist_fig, dist_ax = plt.subplots(figsize=(10, 4))
            dist_ax.bar(unique_years, counts, color='skyblue', alpha=0.7)
            dist_ax.set_xlabel('年份')
            dist_ax.set_ylabel('像元数量')
            dist_ax.set_title('突变年份分布')
            dist_ax.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()

            dist_canvas = FigureCanvasTkAgg(dist_fig, parent)
            dist_canvas.draw()
            dist_canvas.get_tk_widget().pack(pady=10)
            self.current_figures.append(dist_fig)
        else:
            stats_text = "突变统计:\n未检测到显著突变点"

        stats_label = ttk.Label(parent, text=stats_text, justify=LEFT)
        stats_label.pack(pady=5)

        # 添加结果解释
        explanation_text = """
    结果解读指南：
    • 数值：检测到的突变发生年份（如2020、2021等）
    • NaN：无显著突变
    • 突变含义：时间序列中发生显著变化的时刻，可能对应：
      - 自然灾害（火灾、洪水、干旱）
      - 人类活动（砍伐、建设、耕作方式改变）
      - 政策变化（生态保护政策实施）
      - 气候变化影响
    """
        explanation_label = ttk.Label(parent, text=explanation_text, justify=LEFT)
        explanation_label.pack(pady=5)

        # 下载按钮
        ttk.Button(
            parent,
            text="下载 BFAST 突变检测结果 (GeoTIFF)",
            command=lambda: self.download_result(break_da, "bfast_breakpoints.tif"),
            bootstyle=PRIMARY
        ).pack(pady=5)

    # 同时更新批量下载方法，确保包含MK和BFAST
    def add_batch_download_section(self, parent):
        """添加批量下载区域"""
        batch_frame = ttk.LabelFrame(parent, text="📦 批量下载", padding=10)
        batch_frame.pack(fill=X, pady=10)

        # 创建分析结果选择框
        ttk.Label(batch_frame, text="选择要下载的分析结果:").pack(anchor=W)

        self.batch_vars = {}
        batch_check_frame = ttk.Frame(batch_frame)
        batch_check_frame.pack(fill=X, pady=5)

        analyses = [
            ("Theil-Sen 斜率", "theilsen_slope"),
            ("Theil-Sen 截距", "theilsen_intercept"),
            ("Mann-Kendall 检验", "mk"),
            ("BFAST 突变检测", "bfast"),
            ("FFT 振幅", "fft_amp"),
            ("FFT 周期", "fft_period"),
            ("STL 趋势分量", "stl_trend"),
            ("STL 季节分量", "stl_seasonal"),
            ("STL 残差标准差", "stl_resid")
        ]

        # 创建3列的布局
        for i, (name, key) in enumerate(analyses):
            var = tk.BooleanVar(value=True)
            self.batch_vars[key] = var
            cb = ttk.Checkbutton(batch_check_frame, text=name, variable=var)
            cb.grid(row=i // 3, column=i % 3, sticky=W, padx=5, pady=2)

        # 批量下载按钮
        ttk.Button(
            batch_frame,
            text="📥 下载选中结果为ZIP压缩包",
            command=self.batch_download,
            bootstyle=SUCCESS
        ).pack(fill=X, pady=10)

    def batch_download(self):
        """批量下载分析结果"""
        if not self.analysis_results:
            messagebox.showwarning("警告", "没有可下载的分析结果")
            return

        selected_results = {}

        # 收集选中的结果
        for key, var in self.batch_vars.items():
            if var.get():
                if key == "theilsen_slope" and 'theilsen' in self.analysis_results:
                    selected_results["theil_sen_slope"] = self.analysis_results['theilsen']['slope']
                elif key == "theilsen_intercept" and 'theilsen' in self.analysis_results:
                    selected_results["theil_sen_intercept"] = self.analysis_results['theilsen']['intercept']
                elif key == "mk" and 'mk' in self.analysis_results:
                    selected_results["mann_kendall_test"] = self.analysis_results['mk']
                elif key == "bfast" and 'bfast' in self.analysis_results:
                    selected_results["bfast_breakpoints"] = self.analysis_results['bfast']
                elif key == "fft_amp" and 'fft' in self.analysis_results:
                    selected_results["fft_amplitude"] = self.analysis_results['fft']['amplitude']
                elif key == "fft_period" and 'fft' in self.analysis_results:
                    selected_results["fft_period"] = self.analysis_results['fft']['period']
                elif key == "stl_trend" and 'stl' in self.analysis_results:
                    selected_results["stl_trend"] = self.analysis_results['stl']['trend']
                elif key == "stl_seasonal" and 'stl' in self.analysis_results:
                    selected_results["stl_seasonal"] = self.analysis_results['stl']['seasonal']
                elif key == "stl_resid" and 'stl' in self.analysis_results:
                    selected_results["stl_residual"] = self.analysis_results['stl']['resid']

        if not selected_results:
            messagebox.showwarning("警告", "请选择至少一个分析结果进行下载")
            return

        try:
            # 选择保存位置
            file_path = filedialog.asksaveasfilename(
                defaultextension=".zip",
                filetypes=[("ZIP files", "*.zip"), ("All files", "*.*")],
                initialfile="analysis_results.zip"
            )

            if file_path:
                # 创建进度显示
                progress_window = tb.Toplevel(self.root)
                progress_window.title("批量下载")
                progress_window.geometry("300x100")

                progress_label = ttk.Label(progress_window, text="正在生成下载文件...")
                progress_label.pack(pady=10)

                progress_var = tk.DoubleVar()
                progress_bar = ttk.Progressbar(progress_window, variable=progress_var, maximum=100)
                progress_bar.pack(fill=X, padx=20, pady=5)

                def download_thread():
                    try:
                        # 转换数据为TIFF格式
                        tiff_results = {}
                        total = len(selected_results)

                        for i, (name, data_array) in enumerate(selected_results.items()):
                            progress_var.set((i / total) * 100)
                            progress_label.config(text=f"正在处理 {name}...")
                            tiff_data = dataarray_to_bytes_tif(data_array)
                            tiff_results[f"{name}.tif"] = tiff_data

                        # 创建ZIP文件
                        zip_buffer = io.BytesIO()
                        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                            # 添加时间戳文件
                            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                            readme_content = f"""遥感时序分析结果
    生成时间: {timestamp}
    包含的分析结果:
    """
                            for name in selected_results.keys():
                                readme_content += f"- {name}.tif\n"

                            zip_file.writestr("README.txt", readme_content)

                            # 添加各分析结果
                            for name, tiff_data in tiff_results.items():
                                zip_file.writestr(name, tiff_data)

                        zip_buffer.seek(0)

                        # 保存文件
                        with open(file_path, 'wb') as f:
                            f.write(zip_buffer.getvalue())

                        progress_window.destroy()
                        messagebox.showinfo("成功", f"批量下载完成！\n文件已保存: {file_path}")

                    except Exception as e:
                        progress_window.destroy()
                        messagebox.showerror("错误", f"批量下载失败: {e}")

                threading.Thread(target=download_thread, daemon=True).start()

        except Exception as e:
            messagebox.showerror("错误", f"批量下载失败: {e}")

    def add_fft_content(self, results, parent):
        """添加FFT分析结果"""
        amp_da = results['amplitude']
        period_da = results['period']

        # 创建双图表
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        im1 = ax1.imshow(amp_da.values, cmap='hot')
        ax1.set_title("FFT 振幅")
        plt.colorbar(im1, ax=ax1)
        ax1.axis('off')

        im2 = ax2.imshow(period_da.values, cmap='cool')
        ax2.set_title("FFT 主周期")
        plt.colorbar(im2, ax=ax2)
        ax2.axis('off')

        plt.tight_layout()

        # 显示图表
        canvas = FigureCanvasTkAgg(fig, parent)
        canvas.draw()
        canvas.get_tk_widget().pack(pady=10)
        self.current_figures.append(fig)

        # 下载按钮
        btn_frame = ttk.Frame(parent)
        btn_frame.pack(pady=5)

        ttk.Button(
            btn_frame,
            text="下载 FFT 振幅",
            command=lambda: self.download_result(amp_da, "fft_amplitude.tif"),
            bootstyle=PRIMARY
        ).pack(side=LEFT, padx=5)

        ttk.Button(
            btn_frame,
            text="下载 FFT 周期",
            command=lambda: self.download_result(period_da, "fft_period.tif"),
            bootstyle=PRIMARY
        ).pack(side=LEFT, padx=5)

    def add_stl_content(self, results, parent):
        """添加STL分析结果"""
        trend_da = results['trend']
        seasonal_da = results['seasonal']
        resid_da = results['resid']

        # 创建三图表
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

        im1 = ax1.imshow(trend_da.values, cmap='RdYlBu')
        ax1.set_title("STL: 平均趋势分量")
        plt.colorbar(im1, ax=ax1)
        ax1.axis('off')

        im2 = ax2.imshow(seasonal_da.values, cmap='RdYlBu')
        ax2.set_title("STL: 平均季节分量")
        plt.colorbar(im2, ax=ax2)
        ax2.axis('off')

        im3 = ax3.imshow(resid_da.values, cmap='RdYlBu')
        ax3.set_title("STL: 残差标准差")
        plt.colorbar(im3, ax=ax3)
        ax3.axis('off')

        plt.tight_layout()

        # 显示图表
        canvas = FigureCanvasTkAgg(fig, parent)
        canvas.draw()
        canvas.get_tk_widget().pack(pady=10)
        self.current_figures.append(fig)

        # 下载按钮
        btn_frame = ttk.Frame(parent)
        btn_frame.pack(pady=5)

        ttk.Button(
            btn_frame,
            text="下载趋势分量",
            command=lambda: self.download_result(trend_da, "stl_trend_mean.tif"),
            bootstyle=PRIMARY
        ).pack(side=LEFT, padx=2)

        ttk.Button(
            btn_frame,
            text="下载季节分量",
            command=lambda: self.download_result(seasonal_da, "stl_seasonal_mean.tif"),
            bootstyle=PRIMARY
        ).pack(side=LEFT, padx=2)

        ttk.Button(
            btn_frame,
            text="下载残差标准差",
            command=lambda: self.download_result(resid_da, "stl_residual_std.tif"),
            bootstyle=PRIMARY
        ).pack(side=LEFT, padx=2)

    def download_result(self, data_array, filename):
        """下载分析结果"""
        try:
            # 选择保存位置
            file_path = filedialog.asksaveasfilename(
                defaultextension=".tif",
                filetypes=[("TIFF files", "*.tif"), ("All files", "*.*")],
                initialfile=filename
            )

            if file_path:
                # 生成TIFF数据
                tif_data = dataarray_to_bytes_tif(data_array)

                # 保存文件
                with open(file_path, 'wb') as f:
                    f.write(tif_data)

                messagebox.showinfo("成功", f"文件已保存: {file_path}")

        except Exception as e:
            messagebox.showerror("错误", f"下载失败: {e}")

    def on_pixel_change(self, event=None):
        """像元坐标改变时的回调"""
        # 可以在这里实现实时预览
        pass

    def analyze_pixel(self):
        """分析选中像元"""
        if self.data_stack is None:
            messagebox.showwarning("警告", "请先上传数据")
            return

        row = self.row_var.get()
        col = self.col_var.get()

        # 在新线程中执行分析
        def pixel_analysis_thread():
            try:
                # 创建像素分析图表
                fig = self.create_pixel_analysis_figure(row, col)

                # 在UI线程中显示结果
                self.root.after(0, lambda: self.show_pixel_results(fig, row, col))

            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("错误", f"像元分析失败: {e}"))

        threading.Thread(target=pixel_analysis_thread, daemon=True).start()

    def create_pixel_analysis_figure(self, row, col):
        """创建像元分析图表"""
        series = self.data_stack[:, row, col].values
        times = self.data_stack["time"].values

        # 格式化时间标签
        time_labels = []
        for t in times:
            if isinstance(t, np.datetime64):
                time_labels.append(np.datetime_as_string(t, unit='D'))
            else:
                time_labels.append(str(t))

        # 创建图表
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'像元 ({int(row)}, {int(col)}) 时序分析', fontsize=16, fontweight='bold')

        # 原始时序
        ax1.plot(time_labels, series, 'o-', linewidth=2, markersize=4, color='#2E86AB')
        ax1.set_title("原始时序")
        ax1.set_ylabel("值")
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='x', rotation=45)

        # 趋势分析
        valid_mask = ~np.isnan(series)
        if np.sum(valid_mask) >= 3:
            x_numeric = np.arange(len(series))
            valid_x = x_numeric[valid_mask]
            valid_series = series[valid_mask]

            if len(valid_x) >= 2:
                coeffs = np.polyfit(valid_x, valid_series, 1)
                trend_line = np.polyval(coeffs, x_numeric)

                ax2.plot(time_labels, series, 'o-', alpha=0.7)
                ax2.plot(time_labels, trend_line, '--', linewidth=2, color='#A23B72')
                ax2.set_title(f"趋势分析 (斜率: {coeffs[0]:.4f})")
                ax2.set_ylabel("值")
                ax2.grid(True, alpha=0.3)
                ax2.tick_params(axis='x', rotation=45)

        # STL分解尝试
        try:
            from statsmodels.tsa.seasonal import STL

            if np.sum(valid_mask) >= max(3, self.stl_period_var.get() * 2):
                series_filled = series.copy()
                if not np.all(valid_mask):
                    x_numeric = np.arange(len(series))
                    series_filled = np.interp(x_numeric, x_numeric[valid_mask], series[valid_mask])

                stl_result = STL(series_filled, period=self.stl_period_var.get(), robust=True).fit()

                # 趋势分量
                ax3.plot(time_labels, stl_result.trend, linewidth=2, color='#F18F01')
                ax3.set_title("STL趋势分量")
                ax3.set_xlabel("时间")
                ax3.set_ylabel("值")
                ax3.grid(True, alpha=0.3)
                ax3.tick_params(axis='x', rotation=45)

                # 季节分量
                ax4.plot(time_labels, stl_result.seasonal, linewidth=2, color='#C73E1D')
                ax4.set_title("STL季节分量")
                ax4.set_xlabel("时间")
                ax4.set_ylabel("值")
                ax4.grid(True, alpha=0.3)
                ax4.tick_params(axis='x', rotation=45)

        except Exception as e:
            ax3.text(0.5, 0.5, f"STL分析失败\n{e}", ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title("STL分析")
            ax4.text(0.5, 0.5, f"STL分析失败\n{e}", ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title("STL分析")

        plt.tight_layout()
        return fig

    def show_pixel_results(self, fig, row, col):
        """显示像元分析结果"""
        # 创建新窗口显示结果
        result_window = tb.Toplevel(self.root)
        result_window.title(f"像元 ({int(row)}, {int(col)}) 分析结果")
        result_window.geometry("800x600")

        # 显示图表
        canvas = FigureCanvasTkAgg(fig, result_window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=BOTH, expand=True, padx=10, pady=10)

        # 下载按钮
        btn_frame = ttk.Frame(result_window)
        btn_frame.pack(pady=10)

        ttk.Button(
            btn_frame,
            text="下载图表 (PNG)",
            command=lambda: self.download_figure(fig, f"pixel_{int(row)}_{int(col)}.png"),
            bootstyle=PRIMARY
        ).pack(side=LEFT, padx=5)

        self.current_figures.append(fig)

    def download_figure(self, fig, filename):
        """下载图表"""
        try:
            file_path = filedialog.asksaveasfilename(
                defaultextension=".png",
                filetypes=[("PNG files", "*.png"), ("All files", "*.*")],
                initialfile=filename
            )

            if file_path:
                fig.savefig(file_path, bbox_inches='tight', dpi=150)
                messagebox.showinfo("成功", f"图表已保存: {file_path}")

        except Exception as e:
            messagebox.showerror("错误", f"下载失败: {e}")

    def run(self):
        """运行应用程序"""
        self.root.mainloop()


# ==================== 全局变量 ====================

# 全局数据栈变量
data_stack = None

# ==================== 主程序入口 ====================

if __name__ == "__main__":
    app = RemoteSensingApp()
    app.run()
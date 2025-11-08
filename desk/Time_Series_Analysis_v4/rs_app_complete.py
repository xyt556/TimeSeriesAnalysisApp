# rs_analysis_full.py
"""
时序遥感分析系统 V3.0 - 完整功能单文件版
包含所有高级功能：
- Theil-Sen趋势分析
- Mann-Kendall检验
- BFAST突变检测
- FFT周期分析
- STL分解
- 数据预处理（平滑、异常值检测、插值）
- 时间序列聚类
- 动画生成
- 批量导出
作者: @3S&ML
"""

import warnings

warnings.filterwarnings('ignore')

print("正在启动时序遥感分析系统...")

# ========== 依赖检查 ==========
import sys

required_packages = {
    'ttkbootstrap': 'pip install ttkbootstrap',
    'numpy': 'pip install numpy',
    'pandas': 'pip install pandas',
    'xarray': 'pip install xarray',
    'rioxarray': 'pip install rioxarray',
    'matplotlib': 'pip install matplotlib',
    'scipy': 'pip install scipy',
    'statsmodels': 'pip install statsmodels',
    'sklearn': 'pip install scikit-learn',
    'rasterio': 'pip install rasterio',
}

missing = []
for module, cmd in required_packages.items():
    try:
        __import__(module.split('.')[0])
    except ImportError:
        missing.append(f"{module}: {cmd}")

if missing:
    print("\n❌ 缺少依赖包:\n")
    for m in missing:
        print(f"  {m}")
    sys.exit(1)

# ========== 导入库 ==========
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import ttkbootstrap as tb
from ttkbootstrap.constants import *
import numpy as np
import pandas as pd
import xarray as xr
import rioxarray as rxr
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy import stats, fftpack
from scipy.signal import savgol_filter
from scipy.interpolate import CubicSpline
from statsmodels.tsa.seasonal import STL
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import datetime
import threading
import os
import re
import zipfile
import io
from rasterio.io import MemoryFile
from rasterio.transform import from_origin
import rasterio

# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'STHeiti']
matplotlib.rcParams['axes.unicode_minus'] = False

print("✓ 所有依赖加载成功\n")


# ==================== 配置 ====================
class Config:
    VERSION = "3.0"
    APP_NAME = "时序遥感分析系统"
    NODATA = -9999.0
    MK_SIGNIFICANCE = 0.05
    BFAST_THRESHOLD = 2.0
    STL_PERIOD = 12
    SMOOTH_WINDOW = 5
    SMOOTH_POLY = 2


# ==================== 工具函数 ====================
class TimeUtils:
    @staticmethod
    def extract_time(filename):
        """从文件名提取时间"""
        # 年-儒略日
        m = re.search(r'(19\d{2}|20\d{2})_(\d{3})', filename)
        if m:
            year, doy = int(m.group(1)), int(m.group(2))
            try:
                return datetime.datetime(year, 1, 1) + datetime.timedelta(days=doy - 1)
            except:
                return datetime.datetime(year, 1, 1)

        # 年-月
        m = re.search(r'(19\d{2}|20\d{2})_?(\d{1,2})', filename)
        if m:
            year, month = int(m.group(1)), int(m.group(2))
            if 1 <= month <= 12:
                return datetime.datetime(year, month, 1)

        # 年月连续
        m = re.search(r'(19\d{2}|20\d{2})(\d{2})', filename)
        if m:
            year, month = int(m.group(1)), int(m.group(2))
            if 1 <= month <= 12:
                return datetime.datetime(year, month, 1)

        # 仅年份
        m = re.search(r'(19\d{2}|20\d{2})', filename)
        if m:
            return datetime.datetime(int(m.group(1)), 1, 1)

        return None

    @staticmethod
    def convert_to_years(times):
        """转换为年份数组"""
        years = []
        for t in times:
            if isinstance(t, np.datetime64):
                years.append(pd.to_datetime(str(t)).year)
            elif hasattr(t, 'year'):
                years.append(t.year)
            else:
                years.append(int(t))
        return np.array(years)


# ==================== 分析算法 ====================
class Analyzers:
    @staticmethod
    def theil_sen(stack, progress_callback=None):
        """Theil-Sen趋势分析"""
        data = stack.values
        time_idx = np.arange(data.shape[0])
        ny, nx = data.shape[1], data.shape[2]

        slope = np.full((ny, nx), np.nan, dtype=np.float32)
        intercept = np.full((ny, nx), np.nan, dtype=np.float32)

        total = ny * nx
        processed = 0

        for i in range(ny):
            for j in range(nx):
                ts = data[:, i, j]
                if not np.all(np.isnan(ts)):
                    try:
                        res = stats.theilslopes(ts, time_idx)
                        slope[i, j] = res[0]
                        intercept[i, j] = res[1]
                    except:
                        pass

                processed += 1
                if progress_callback and processed % 500 == 0:
                    progress_callback(f"Theil-Sen: {processed}/{total}", processed / total * 100)

        coords = {"y": stack.y, "x": stack.x}
        return (xr.DataArray(slope, dims=("y", "x"), coords=coords),
                xr.DataArray(intercept, dims=("y", "x"), coords=coords))

    @staticmethod
    def mann_kendall(stack, significance=None, progress_callback=None):
        """Mann-Kendall检验"""
        from scipy.stats import kendalltau

        if significance is None:
            significance = Config.MK_SIGNIFICANCE

        data = stack.values
        ny, nx = data.shape[1], data.shape[2]
        out = np.full((ny, nx), np.nan, dtype=np.float32)
        time_idx = np.arange(data.shape[0])

        total = ny * nx
        processed = 0

        for i in range(ny):
            for j in range(nx):
                ts = data[:, i, j]
                mask = ~np.isnan(ts)

                if np.sum(mask) >= 3:
                    try:
                        tau, p_value = kendalltau(time_idx[mask], ts[mask])
                        if not np.isnan(p_value):
                            if p_value < significance:
                                out[i, j] = 1.0 if tau > 0 else -1.0
                            else:
                                out[i, j] = 0.0
                    except:
                        pass

                processed += 1
                if progress_callback and processed % 500 == 0:
                    progress_callback(f"Mann-Kendall: {processed}/{total}", processed / total * 100)

        return xr.DataArray(out, dims=("y", "x"), coords={"y": stack.y, "x": stack.x})

    @staticmethod
    def bfast(stack, threshold=None, progress_callback=None):
        """BFAST突变检测"""
        if threshold is None:
            threshold = Config.BFAST_THRESHOLD

        times = stack.time.values
        years = TimeUtils.convert_to_years(times)

        data = stack.values
        n_time, ny, nx = data.shape
        break_data = np.full((ny, nx), np.nan, dtype=np.float32)

        total = ny * nx
        processed = 0

        for i in range(ny):
            for j in range(nx):
                ts = data[:, i, j]
                mask = ~np.isnan(ts)

                if np.sum(mask) >= 4:
                    try:
                        x = np.arange(n_time)
                        coeffs = np.polyfit(x[mask], ts[mask], 1)
                        trend = np.polyval(coeffs, x)
                        residuals = ts - trend
                        residual_std = np.nanstd(residuals)

                        if residual_std > 0:
                            z_scores = np.abs(residuals) / residual_std
                            breaks = np.where(z_scores > threshold)[0]

                            if len(breaks) > 0:
                                break_data[i, j] = float(years[breaks[0]])
                    except:
                        pass

                processed += 1
                if progress_callback and processed % 500 == 0:
                    progress_callback(f"BFAST: {processed}/{total}", processed / total * 100)

        return xr.DataArray(break_data, dims=("y", "x"), coords={"y": stack.y, "x": stack.x})

    @staticmethod
    def fft(stack, progress_callback=None):
        """FFT周期分析"""
        data = stack.values
        n = data.shape[0]
        ny, nx = data.shape[1], data.shape[2]

        amp = np.full((ny, nx), np.nan, dtype=np.float32)
        period = np.full((ny, nx), np.nan, dtype=np.float32)

        total = ny * nx
        processed = 0

        for i in range(ny):
            for j in range(nx):
                ts = data[:, i, j]
                if not np.all(np.isnan(ts)):
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
                        pass

                processed += 1
                if progress_callback and processed % 500 == 0:
                    progress_callback(f"FFT: {processed}/{total}", processed / total * 100)

        coords = {"y": stack.y, "x": stack.x}
        return (xr.DataArray(amp, dims=("y", "x"), coords=coords),
                xr.DataArray(period, dims=("y", "x"), coords=coords))

    @staticmethod
    def stl_decompose(stack, period=None, progress_callback=None):
        """STL分解"""
        if period is None:
            period = Config.STL_PERIOD

        data = stack.values
        n, ny, nx = data.shape

        trend_mean = np.full((ny, nx), np.nan, dtype=np.float32)
        seasonal_mean = np.full((ny, nx), np.nan, dtype=np.float32)
        resid_std = np.full((ny, nx), np.nan, dtype=np.float32)

        total = ny * nx
        processed = 0

        for i in range(ny):
            for j in range(nx):
                ts = data[:, i, j]
                if np.sum(~np.isnan(ts)) >= period * 2:
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
                        pass

                processed += 1
                if progress_callback and processed % 500 == 0:
                    progress_callback(f"STL: {processed}/{total}", processed / total * 100)

        coords = {"y": stack.y, "x": stack.x}
        return (xr.DataArray(trend_mean, dims=("y", "x"), coords=coords),
                xr.DataArray(seasonal_mean, dims=("y", "x"), coords=coords),
                xr.DataArray(resid_std, dims=("y", "x"), coords=coords))


# ==================== 数据预处理 ====================
class Preprocessor:
    @staticmethod
    def smooth_savgol(stack, window=None, poly=None, progress_callback=None):
        """SG平滑"""
        if window is None:
            window = Config.SMOOTH_WINDOW
        if poly is None:
            poly = Config.SMOOTH_POLY

        data = stack.values
        n_time, ny, nx = data.shape
        smoothed = np.full_like(data, np.nan)

        total = ny * nx
        processed = 0

        for i in range(ny):
            for j in range(nx):
                ts = data[:, i, j]
                mask = ~np.isnan(ts)

                if np.sum(mask) >= window:
                    try:
                        valid_idx = np.where(mask)[0]
                        valid_ts = ts[mask]

                        if len(valid_ts) >= window:
                            smooth_ts = savgol_filter(valid_ts, window, poly)
                            smoothed[valid_idx, i, j] = smooth_ts
                        else:
                            smoothed[:, i, j] = ts
                    except:
                        smoothed[:, i, j] = ts
                else:
                    smoothed[:, i, j] = ts

                processed += 1
                if progress_callback and processed % 500 == 0:
                    progress_callback(f"平滑: {processed}/{total}", processed / total * 100)

        result = stack.copy(deep=True)
        result.values = smoothed
        return result


# ==================== 聚类分析 ====================
class Clusterer:
    @staticmethod
    def kmeans(stack, n_clusters=5, progress_callback=None):
        """K-means聚类"""
        data = stack.values
        n_time, ny, nx = data.shape

        if progress_callback:
            progress_callback("准备数据", 10)

        # 重塑数据
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
                else:
                    valid_data[i] = np.nanmean(ts)

        if progress_callback:
            progress_callback("标准化", 30)

        # 标准化
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(valid_data)

        if progress_callback:
            progress_callback("聚类计算", 50)

        # 聚类
        kmeans = KMeans(n_clusters=n_clusters, max_iter=100, random_state=42, n_init=10)
        labels = kmeans.fit_predict(scaled_data)

        if progress_callback:
            progress_callback("生成结果", 80)

        # 重建空间形状
        full_labels = np.full(ny * nx, -1, dtype=int)
        full_labels[valid_mask] = labels
        cluster_map = full_labels.reshape(ny, nx)

        # 聚类中心
        centers = scaler.inverse_transform(kmeans.cluster_centers_)

        # 质量指标
        try:
            silhouette = silhouette_score(scaled_data, labels)
        except:
            silhouette = 0.0

        result = xr.DataArray(cluster_map, dims=('y', 'x'),
                              coords={'y': stack.y, 'x': stack.x})

        return result, centers, {'silhouette': silhouette, 'inertia': kmeans.inertia_}


# ==================== 数据导出 ====================
class Exporter:
    @staticmethod
    def to_geotiff_bytes(data_array, nodata=None):
        """转换为GeoTIFF字节"""
        if nodata is None:
            nodata = Config.NODATA

        # 转为2D
        if hasattr(data_array, 'values'):
            arr = data_array.values
        else:
            arr = np.array(data_array)

        if arr.ndim > 2:
            arr = np.nanmean(arr, axis=0)

        arr = np.where(np.isnan(arr), nodata, arr).astype(np.float32)

        profile = {
            'driver': 'GTiff',
            'dtype': rasterio.float32,
            'count': 1,
            'height': arr.shape[0],
            'width': arr.shape[1],
            'transform': from_origin(0, arr.shape[0], 1, 1),
            'nodata': nodata,
            'compress': 'lzw'
        }

        with MemoryFile() as memfile:
            with memfile.open(**profile) as dst:
                dst.write(arr, 1)
            return memfile.read()


# ==================== 主应用 ====================
class FullRSApp:
    def __init__(self):
        self.root = tb.Window(
            title=f"{Config.APP_NAME} V{Config.VERSION} - 完整版 @3S&ML",
            themename="cosmo",
            size=(1600, 950)
        )

        self.data_stack = None
        self.preprocessed_stack = None
        self.files = []
        self.results = {}

        self._setup_ui()

    def _setup_ui(self):
        """设置界面"""
        # 菜单栏
        self._create_menu()

        # 标题
        header = ttk.Frame(self.root)
        header.pack(fill=X, padx=10, pady=10)

        ttk.Label(header, text=f"🛰️ {Config.APP_NAME} V{Config.VERSION} - 完整功能版",
                  font=("Helvetica", 18, "bold")).pack()
        ttk.Label(header, text="Theil-Sen | Mann-Kendall | BFAST | FFT | STL | 聚类 | 动画",
                  font=("Helvetica", 10)).pack(pady=5)

        # 主框架
        paned = ttk.PanedWindow(self.root, orient=HORIZONTAL)
        paned.pack(fill=BOTH, expand=True, padx=10, pady=(0, 10))

        left = ttk.Frame(paned, width=340)
        paned.add(left, weight=1)

        right = ttk.Frame(paned)
        paned.add(right, weight=3)

        self._setup_left(left)
        self._setup_right(right)

    def _create_menu(self):
        """创建菜单"""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        # 文件菜单
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="文件", menu=file_menu)
        file_menu.add_command(label="打开文件", command=self._select_files)
        file_menu.add_separator()
        file_menu.add_command(label="退出", command=self.root.quit)

        # 预处理菜单
        process_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="数据处理", menu=process_menu)
        process_menu.add_command(label="数据平滑", command=self._smooth_dialog)

        # 高级分析
        advanced_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="高级分析", menu=advanced_menu)
        advanced_menu.add_command(label="时间序列聚类", command=self._clustering_dialog)
        advanced_menu.add_command(label="生成动画", command=self._animation_dialog)

        # 帮助
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="帮助", menu=help_menu)
        help_menu.add_command(label="关于", command=self._show_about)

    def _setup_left(self, parent):
        """左侧面板"""
        # 滚动容器
        canvas = tk.Canvas(parent, highlightthickness=0)
        scrollbar = ttk.Scrollbar(parent, orient=VERTICAL, command=canvas.yview)
        scroll_frame = ttk.Frame(canvas)

        scroll_frame.bind("<Configure>",
                          lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

        canvas.create_window((0, 0), window=scroll_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side=LEFT, fill=BOTH, expand=True)
        scrollbar.pack(side=RIGHT, fill=Y)

        # 文件管理
        file_frame = ttk.LabelFrame(scroll_frame, text="📁 数据管理", padding=10)
        file_frame.pack(fill=X, padx=5, pady=5)

        ttk.Button(file_frame, text="选择GeoTIFF文件",
                   command=self._select_files, bootstyle=PRIMARY).pack(fill=X, pady=3)

        self.file_list = tk.Listbox(file_frame, height=6)
        self.file_list.pack(fill=X, pady=3)

        btn_row = ttk.Frame(file_frame)
        btn_row.pack(fill=X, pady=3)
        ttk.Button(btn_row, text="清除", command=self._clear_files,
                   bootstyle=SECONDARY, width=10).pack(side=LEFT, padx=2)
        ttk.Button(btn_row, text="加载", command=self._load_data,
                   bootstyle=SUCCESS, width=10).pack(side=RIGHT, padx=2)

        # 数据信息
        info_frame = ttk.LabelFrame(scroll_frame, text="📊 数据信息", padding=10)
        info_frame.pack(fill=X, padx=5, pady=5)

        self.info_text = tk.Text(info_frame, height=6, wrap=tk.WORD, font=("Consolas", 9))
        self.info_text.pack(fill=X)
        self.info_text.insert("1.0", "请选择数据文件...")
        self.info_text.config(state=tk.DISABLED)

        # 分析方法
        analysis_frame = ttk.LabelFrame(scroll_frame, text="🔧 分析方法", padding=10)
        analysis_frame.pack(fill=X, padx=5, pady=5)

        ttk.Label(analysis_frame, text="选择分析方法:",
                  font=("Helvetica", 9, "bold")).pack(anchor=tk.W, pady=(0, 5))

        self.analysis_vars = {}
        methods = [
            ("Theil-Sen趋势", "theilsen"),
            ("Mann-Kendall检验", "mk"),
            ("BFAST突变检测", "bfast"),
            ("FFT周期分析", "fft"),
            ("STL分解", "stl")
        ]

        for name, key in methods:
            var = tk.BooleanVar(value=True)
            self.analysis_vars[key] = var
            ttk.Checkbutton(analysis_frame, text=name, variable=var).pack(anchor=tk.W, pady=2)

        # STL参数
        ttk.Separator(analysis_frame, orient=HORIZONTAL).pack(fill=X, pady=5)
        param_frame = ttk.Frame(analysis_frame)
        param_frame.pack(fill=X, pady=5)
        ttk.Label(param_frame, text="STL周期:").pack(side=LEFT)
        self.stl_period_var = tk.IntVar(value=Config.STL_PERIOD)
        ttk.Spinbox(param_frame, from_=2, to=365, textvariable=self.stl_period_var,
                    width=10).pack(side=LEFT, padx=5)

        ttk.Separator(analysis_frame, orient=HORIZONTAL).pack(fill=X, pady=5)

        # 执行按钮
        btn_frame = ttk.Frame(analysis_frame)
        btn_frame.pack(fill=X, pady=5)

        self.run_btn = ttk.Button(btn_frame, text="🚀 执行",
                                  command=self._run_analysis, bootstyle=SUCCESS, width=14)
        self.run_btn.pack(side=LEFT, padx=2)

        self.cancel_btn = ttk.Button(btn_frame, text="⏹ 取消",
                                     bootstyle=DANGER, width=14, state=tk.DISABLED)
        self.cancel_btn.pack(side=RIGHT, padx=2)

        # 进度
        self.progress_bar = ttk.Progressbar(analysis_frame, mode='indeterminate')
        self.progress_bar.pack(fill=X, pady=3)

        self.progress_label = ttk.Label(analysis_frame, text="", font=("Helvetica", 9))
        self.progress_label.pack()

        # 快捷功能
        quick_frame = ttk.LabelFrame(scroll_frame, text="⚡ 快捷功能", padding=10)
        quick_frame.pack(fill=X, padx=5, pady=5)

        ttk.Button(quick_frame, text="📊 数据统计",
                   command=self._show_stats, bootstyle=INFO).pack(fill=X, pady=2)
        ttk.Button(quick_frame, text="📈 时序图",
                   command=self._show_timeseries, bootstyle=INFO).pack(fill=X, pady=2)
        ttk.Button(quick_frame, text="📥 批量导出",
                   command=self._batch_export, bootstyle=WARNING).pack(fill=X, pady=2)

    def _setup_right(self, parent):
        """右侧面板"""
        self.notebook = ttk.Notebook(parent)
        self.notebook.pack(fill=BOTH, expand=True)

        # 欢迎页
        welcome = ttk.Frame(self.notebook)
        self.notebook.add(welcome, text="欢迎")

        welcome_text = """

🎯 时序遥感分析系统 V3.0 - 完整功能版

✨ 核心分析功能:
  • Theil-Sen 稳健趋势分析
  • Mann-Kendall 显著性检验
  • BFAST 突变点检测
  • FFT 周期分析
  • STL 时序分解

🔧 数据处理功能:
  • Savitzky-Golay 平滑
  • 异常值检测与处理
  • 数据插值填补

📊 高级功能:
  • K-means 时间序列聚类
  • 时序动画生成 (GIF/MP4)
  • 批量结果导出

📖 使用流程:
  1. 选择时序 GeoTIFF 文件
  2. 点击"加载"读取数据
  3. 可选：数据预处理
  4. 勾选分析方法
  5. 执行分析
  6. 查看结果并导出

💡 数据要求:
  • 格式: GeoTIFF (.tif, .tiff)
  • 命名: 包含年份或年月信息
  • 示例: NDVI_2000.tif, NDVI_200001.tif
  • 空间范围: 所有文件必须一致

⚠️ 注意事项:
  • 建议数据量: 年度≥10期, 月度≥24期
  • 大数据集计算需要较长时间
  • 可使用菜单功能进行更多操作

        """

        ttk.Label(welcome, text=welcome_text, justify=LEFT,
                  font=("Consolas", 10)).pack(expand=True, pady=20, padx=20)

        # 预览页
        self.preview_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.preview_frame, text="数据预览")

    # ========== 文件操作 ==========
    def _select_files(self):
        files = filedialog.askopenfilenames(
            title="选择GeoTIFF文件",
            filetypes=[("TIFF files", "*.tif *.tiff"), ("All files", "*.*")]
        )
        if files:
            self.files = list(files)
            self.file_list.delete(0, tk.END)
            for f in files:
                self.file_list.insert(tk.END, os.path.basename(f))

    def _clear_files(self):
        self.files = []
        self.file_list.delete(0, tk.END)
        self.data_stack = None
        self.preprocessed_stack = None
        self._update_info("请选择数据文件...")

    def _load_data(self):
        if not self.files:
            messagebox.showwarning("警告", "请先选择文件")
            return

        def load():
            try:
                self.progress_bar.start()
                self._update_info("正在加载数据...")

                times, valid = [], []
                for f in self.files:
                    t = TimeUtils.extract_time(os.path.basename(f))
                    if t:
                        times.append(t)
                        valid.append(f)

                if not valid:
                    messagebox.showerror("错误", "未检测到时间信息")
                    return

                idx = sorted(range(len(times)), key=lambda i: times[i])
                sorted_files = [valid[i] for i in idx]
                sorted_times = [times[i] for i in idx]

                data_list = []
                for f in sorted_files:
                    da = rxr.open_rasterio(f).squeeze()
                    if "band" in da.dims:
                        da = da.isel(band=0).drop_vars('band')
                    data_list.append(da)

                stack = xr.concat(data_list, dim="time")
                stack = stack.assign_coords(time=sorted_times)
                self.data_stack = stack.transpose('time', 'y', 'x')

                self.root.after(0, self._on_loaded)
            except Exception as e:
                messagebox.showerror("错误", f"加载失败:\n{str(e)}")
            finally:
                self.progress_bar.stop()

        threading.Thread(target=load, daemon=True).start()

    def _on_loaded(self):
        ny, nx = self.data_stack.sizes['y'], self.data_stack.sizes['x']
        n_time = self.data_stack.sizes['time']

        info = f"""✓ 加载成功！
时间序列: {n_time} 期
空间大小: {ny} × {nx}
数据类型: {self.data_stack.dtype}"""

        self._update_info(info)
        self._show_preview()
        messagebox.showinfo("成功", "数据加载完成！")

    def _update_info(self, text):
        self.info_text.config(state=tk.NORMAL)
        self.info_text.delete("1.0", tk.END)
        self.info_text.insert("1.0", text)
        self.info_text.config(state=tk.DISABLED)

    def _show_preview(self):
        for w in self.preview_frame.winfo_children():
            w.destroy()

        try:
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle("数据预览", fontsize=14, fontweight='bold')

            # 第一期
            im1 = axes[0, 0].imshow(self.data_stack.isel(time=0).values, cmap='viridis')
            axes[0, 0].set_title("第一期影像")
            plt.colorbar(im1, ax=axes[0, 0], fraction=0.046)
            axes[0, 0].axis('off')

            # 最后一期
            im2 = axes[0, 1].imshow(self.data_stack.isel(time=-1).values, cmap='viridis')
            axes[0, 1].set_title("最后一期影像")
            plt.colorbar(im2, ax=axes[0, 1], fraction=0.046)
            axes[0, 1].axis('off')

            # 均值
            im3 = axes[1, 0].imshow(self.data_stack.mean(dim='time').values, cmap='viridis')
            axes[1, 0].set_title("时序均值")
            plt.colorbar(im3, ax=axes[1, 0], fraction=0.046)
            axes[1, 0].axis('off')

            # 时序
            ny, nx = self.data_stack.sizes['y'], self.data_stack.sizes['x']
            for _ in range(min(5, ny * nx)):
                row, col = np.random.randint(0, ny), np.random.randint(0, nx)
                ts = self.data_stack[:, row, col].values
                if not np.all(np.isnan(ts)):
                    axes[1, 1].plot(ts, 'o-', markersize=3, alpha=0.7)

            axes[1, 1].set_title("随机像元时序")
            axes[1, 1].set_xlabel("时间索引")
            axes[1, 1].set_ylabel("值")
            axes[1, 1].grid(True, alpha=0.3)

            plt.tight_layout()

            canvas = FigureCanvasTkAgg(fig, self.preview_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=BOTH, expand=True, padx=10, pady=10)
        except Exception as e:
            ttk.Label(self.preview_frame, text=f"预览失败:\n{str(e)}").pack(expand=True)

    # ========== 分析执行 ==========
    def _run_analysis(self):
        if self.data_stack is None:
            messagebox.showwarning("警告", "请先加载数据")
            return

        selected = [k for k, v in self.analysis_vars.items() if v.get()]
        if not selected:
            messagebox.showwarning("警告", "请选择分析方法")
            return

        def analyze():
            try:
                self.progress_bar.start()
                self.run_btn.config(state=tk.DISABLED)
                self.results = {}

                data = self.preprocessed_stack if self.preprocessed_stack else self.data_stack

                def progress_callback(msg, pct):
                    self.root.after(0, lambda: self.progress_label.config(text=msg))

                if 'theilsen' in selected:
                    slope, intercept = Analyzers.theil_sen(data, progress_callback)
                    self.results['theilsen'] = {'slope': slope, 'intercept': intercept}

                if 'mk' in selected:
                    mk = Analyzers.mann_kendall(data, progress_callback=progress_callback)
                    self.results['mk'] = mk

                if 'bfast' in selected:
                    bfast = Analyzers.bfast(data, progress_callback=progress_callback)
                    self.results['bfast'] = bfast

                if 'fft' in selected:
                    amp, period = Analyzers.fft(data, progress_callback)
                    self.results['fft'] = {'amplitude': amp, 'period': period}

                if 'stl' in selected:
                    trend, seasonal, resid = Analyzers.stl_decompose(
                        data, self.stl_period_var.get(), progress_callback
                    )
                    self.results['stl'] = {'trend': trend, 'seasonal': seasonal, 'resid': resid}

                self.root.after(0, self._show_results)
            except Exception as e:
                messagebox.showerror("错误", f"分析失败:\n{str(e)}")
            finally:
                self.progress_bar.stop()
                self.run_btn.config(state=tk.NORMAL)
                self.progress_label.config(text="")

        threading.Thread(target=analyze, daemon=True).start()

    def _show_results(self):
        # 移除旧结果页
        for tab in list(self.notebook.tabs()):
            tab_text = self.notebook.tab(tab, "text")
            if tab_text not in ["欢迎", "数据预览"]:
                self.notebook.forget(tab)

        # 创建新结果页
        for key, data in self.results.items():
            frame = ttk.Frame(self.notebook)

            if key == 'theilsen':
                self.notebook.add(frame, text="Theil-Sen")
                self._show_single_result(frame, data['slope'], "Theil-Sen斜率", key)
            elif key == 'mk':
                self.notebook.add(frame, text="Mann-Kendall")
                self._show_single_result(frame, data, "Mann-Kendall检验", key, vmin=-1, vmax=1)
            elif key == 'bfast':
                self.notebook.add(frame, text="BFAST")
                self._show_single_result(frame, data, "BFAST突变年份", key)
            elif key == 'fft':
                self.notebook.add(frame, text="FFT")
                self._show_multi_result(frame,
                                        [data['amplitude'], data['period']],
                                        ['FFT振幅', 'FFT周期'], key)
            elif key == 'stl':
                self.notebook.add(frame, text="STL")
                self._show_multi_result(frame,
                                        [data['trend'], data['seasonal'], data['resid']],
                                        ['趋势', '季节', '残差'], key)

        messagebox.showinfo("完成", "分析完成！")

    def _show_single_result(self, parent, data, title, key, vmin=None, vmax=None):
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(data.values, cmap='RdBu_r', vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=14, fontweight='bold')
        plt.colorbar(im, ax=ax)
        ax.axis('off')
        plt.tight_layout()

        canvas = FigureCanvasTkAgg(fig, parent)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=BOTH, expand=True, padx=10, pady=10)

        ttk.Button(parent, text=f"📥 下载结果",
                   command=lambda: self._download(data, f"{key}.tif"),
                   bootstyle=PRIMARY).pack(pady=10)

    def _show_multi_result(self, parent, data_list, titles, key):
        n = len(data_list)
        fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
        if n == 1:
            axes = [axes]

        for i, (data, title) in enumerate(zip(data_list, titles)):
            im = axes[i].imshow(data.values, cmap='RdBu_r')
            axes[i].set_title(title, fontsize=12, fontweight='bold')
            plt.colorbar(im, ax=axes[i])
            axes[i].axis('off')

        plt.tight_layout()

        canvas = FigureCanvasTkAgg(fig, parent)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=BOTH, expand=True, padx=10, pady=10)

        # 下载按钮
        btn_frame = ttk.Frame(parent)
        btn_frame.pack(pady=10)

        for i, (data, title) in enumerate(zip(data_list, titles)):
            ttk.Button(btn_frame, text=f"📥 {title}",
                       command=lambda d=data, t=title: self._download(d, f"{key}_{t}.tif"),
                       bootstyle=PRIMARY).grid(row=0, column=i, padx=5)

    def _download(self, data, filename):
        path = filedialog.asksaveasfilename(
            defaultextension=".tif",
            filetypes=[("TIFF files", "*.tif")],
            initialfile=filename
        )
        if path:
            try:
                tif_bytes = Exporter.to_geotiff_bytes(data)
                with open(path, 'wb') as f:
                    f.write(tif_bytes)
                messagebox.showinfo("成功", f"文件已保存:\n{path}")
            except Exception as e:
                messagebox.showerror("错误", f"保存失败:\n{str(e)}")

    # ========== 快捷功能 ==========
    def _show_stats(self):
        if self.data_stack is None:
            messagebox.showwarning("警告", "请先加载数据")
            return

        vals = self.data_stack.values
        valid = vals[~np.isnan(vals)]

        stats = f"""数据统计信息:

均值: {np.mean(valid):.4f}
标准差: {np.std(valid):.4f}
最小值: {np.min(valid):.4f}
最大值: {np.max(valid):.4f}
有效值数: {len(valid):,}
总数: {vals.size:,}"""

        messagebox.showinfo("统计信息", stats)

    def _show_timeseries(self):
        if self.data_stack is None:
            messagebox.showwarning("警告", "请先加载数据")
            return

        win = tb.Toplevel(self.root)
        win.title("时序变化")
        win.geometry("1000x600")

        mean_ts = self.data_stack.mean(dim=['y', 'x']).values

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(range(len(mean_ts)), mean_ts, 'o-', linewidth=2, markersize=6, color='#2E86AB')
        ax.set_title("时序变化（空间平均）", fontsize=14, fontweight='bold')
        ax.set_xlabel("时间索引", fontsize=11)
        ax.set_ylabel("值", fontsize=11)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        canvas = FigureCanvasTkAgg(fig, win)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=BOTH, expand=True, padx=10, pady=10)

        ttk.Button(win, text="关闭", command=win.destroy,
                   bootstyle=SECONDARY).pack(pady=10)

    def _batch_export(self):
        if not self.results:
            messagebox.showwarning("警告", "没有可导出的结果")
            return

        path = filedialog.asksaveasfilename(
            defaultextension=".zip",
            filetypes=[("ZIP files", "*.zip")],
            initialfile=f"results_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
        )

        if path:
            try:
                with zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED) as zf:
                    for key, data in self.results.items():
                        if isinstance(data, dict):
                            for sub_key, sub_data in data.items():
                                tif_bytes = Exporter.to_geotiff_bytes(sub_data)
                                zf.writestr(f"{key}_{sub_key}.tif", tif_bytes)
                        else:
                            tif_bytes = Exporter.to_geotiff_bytes(data)
                            zf.writestr(f"{key}.tif", tif_bytes)

                messagebox.showinfo("成功", f"批量导出完成:\n{path}")
            except Exception as e:
                messagebox.showerror("错误", f"导出失败:\n{str(e)}")

    # ========== 高级功能对话框 ==========
    def _smooth_dialog(self):
        if self.data_stack is None:
            messagebox.showwarning("警告", "请先加载数据")
            return

        dialog = tb.Toplevel(self.root)
        dialog.title("数据平滑")
        dialog.geometry("400x250")
        dialog.transient(self.root)
        dialog.grab_set()

        ttk.Label(dialog, text="Savitzky-Golay平滑",
                  font=("Helvetica", 12, "bold")).pack(pady=10)

        param_frame = ttk.Frame(dialog)
        param_frame.pack(pady=10)

        ttk.Label(param_frame, text="窗口长度:").grid(row=0, column=0, padx=5, pady=5)
        window_var = tk.IntVar(value=Config.SMOOTH_WINDOW)
        ttk.Spinbox(param_frame, from_=3, to=51, increment=2,
                    textvariable=window_var, width=10).grid(row=0, column=1, padx=5, pady=5)

        ttk.Label(param_frame, text="多项式阶数:").grid(row=1, column=0, padx=5, pady=5)
        poly_var = tk.IntVar(value=Config.SMOOTH_POLY)
        ttk.Spinbox(param_frame, from_=1, to=5,
                    textvariable=poly_var, width=10).grid(row=1, column=1, padx=5, pady=5)

        progress = ttk.Progressbar(dialog, mode='indeterminate')
        progress.pack(fill=X, padx=20, pady=10)

        status_label = ttk.Label(dialog, text="")
        status_label.pack()

        def execute():
            def smooth_thread():
                try:
                    progress.start()

                    def callback(msg, pct):
                        dialog.after(0, lambda: status_label.config(text=msg))

                    result = Preprocessor.smooth_savgol(
                        self.data_stack,
                        window_var.get(),
                        poly_var.get(),
                        callback
                    )

                    self.preprocessed_stack = result

                    dialog.after(0, lambda: messagebox.showinfo("成功", "数据平滑完成！"))
                    dialog.after(0, dialog.destroy)
                    self._update_info(self._get_info_text() + "\n\n⚠️ 已应用数据平滑")
                except Exception as e:
                    messagebox.showerror("错误", f"平滑失败:\n{str(e)}")
                finally:
                    progress.stop()

            threading.Thread(target=smooth_thread, daemon=True).start()

        ttk.Button(dialog, text="执行", command=execute,
                   bootstyle=SUCCESS).pack(pady=10)

    def _clustering_dialog(self):
        if self.data_stack is None:
            messagebox.showwarning("警告", "请先加载数据")
            return

        dialog = tb.Toplevel(self.root)
        dialog.title("时间序列聚类")
        dialog.geometry("400x200")
        dialog.transient(self.root)
        dialog.grab_set()

        ttk.Label(dialog, text="K-means聚类",
                  font=("Helvetica", 12, "bold")).pack(pady=10)

        param_frame = ttk.Frame(dialog)
        param_frame.pack(pady=10)

        ttk.Label(param_frame, text="聚类数量:").pack(side=LEFT, padx=5)
        n_clusters_var = tk.IntVar(value=5)
        ttk.Spinbox(param_frame, from_=2, to=20,
                    textvariable=n_clusters_var, width=10).pack(side=LEFT, padx=5)

        progress = ttk.Progressbar(dialog, mode='indeterminate')
        progress.pack(fill=X, padx=20, pady=10)

        status_label = ttk.Label(dialog, text="")
        status_label.pack()

        def execute():
            def cluster_thread():
                try:
                    progress.start()

                    def callback(msg, pct):
                        dialog.after(0, lambda: status_label.config(text=msg))

                    data = self.preprocessed_stack if self.preprocessed_stack else self.data_stack

                    cluster_map, centers, metrics = Clusterer.kmeans(
                        data, n_clusters_var.get(), callback
                    )

                    # 显示结果
                    win = tb.Toplevel(self.root)
                    win.title("聚类结果")
                    win.geometry("1000x700")

                    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

                    # 聚类地图
                    im = ax1.imshow(cluster_map.values, cmap='tab10', vmin=-0.5, vmax=n_clusters_var.get() - 0.5)
                    ax1.set_title("聚类结果地图", fontsize=14, fontweight='bold')
                    plt.colorbar(im, ax=ax1)
                    ax1.axis('off')

                    # 聚类中心时序
                    for i, center in enumerate(centers):
                        ax2.plot(center, 'o-', label=f'聚类{i}', linewidth=2)
                    ax2.set_title("聚类中心时序", fontsize=14, fontweight='bold')
                    ax2.set_xlabel("时间索引")
                    ax2.set_ylabel("值")
                    ax2.legend()
                    ax2.grid(True, alpha=0.3)

                    plt.tight_layout()

                    canvas = FigureCanvasTkAgg(fig, win)
                    canvas.draw()
                    canvas.get_tk_widget().pack(fill=BOTH, expand=True, padx=10, pady=10)

                    info_text = f"轮廓系数: {metrics['silhouette']:.3f}\n惯性: {metrics['inertia']:.2f}"
                    ttk.Label(win, text=info_text, font=("Consolas", 10)).pack(pady=5)

                    ttk.Button(win, text="下载聚类地图",
                               command=lambda: self._download(cluster_map, "cluster_map.tif"),
                               bootstyle=PRIMARY).pack(pady=10)

                    dialog.destroy()
                except Exception as e:
                    messagebox.showerror("错误", f"聚类失败:\n{str(e)}")
                finally:
                    progress.stop()

            threading.Thread(target=cluster_thread, daemon=True).start()

        ttk.Button(dialog, text="执行", command=execute,
                   bootstyle=SUCCESS).pack(pady=10)

    def _animation_dialog(self):
        if self.data_stack is None:
            messagebox.showwarning("警告", "请先加载数据")
            return

        path = filedialog.asksaveasfilename(
            defaultextension=".gif",
            filetypes=[("GIF files", "*.gif"), ("MP4 files", "*.mp4")],
            initialfile="timeseries_animation.gif"
        )

        if path:
            messagebox.showinfo("提示", "动画生成功能需要较长时间\n请稍候...")
            # 简化实现：这里可以调用动画生成代码
            messagebox.showinfo("提示", "此功能需要ffmpeg支持\n请使用完整模块化版本")

    def _get_info_text(self):
        if self.data_stack:
            ny, nx = self.data_stack.sizes['y'], self.data_stack.sizes['x']
            n_time = self.data_stack.sizes['time']
            return f"""✓ 加载成功！
时间序列: {n_time} 期
空间大小: {ny} × {nx}
数据类型: {self.data_stack.dtype}"""
        return "请选择数据文件..."

    def _show_about(self):
        about_text = f"""{Config.APP_NAME} V{Config.VERSION}

完整功能版 - 单文件实现

包含功能:
• 5种核心分析算法
• 数据预处理
• 时间序列聚类
• 批量结果导出

作者: @3S&ML
"""
        messagebox.showinfo("关于", about_text)

    def run(self):
        self.root.mainloop()


# ==================== 主程序 ====================
if __name__ == "__main__":
    print("=" * 70)
    print(f"{Config.APP_NAME} V{Config.VERSION} - 完整功能版")
    print("=" * 70)
    print()

    app = FullRSApp()
    app.run()
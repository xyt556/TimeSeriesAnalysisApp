# rs_step1_base.py
"""
时序遥感分析系统 - 第一步：基础框架
功能：数据加载、预览、基础统计

下一步将添加：Theil-Sen + Mann-Kendall分析
"""

import warnings

warnings.filterwarnings('ignore')

print("正在启动...")

# ========== 依赖检查 ==========
import sys

required = {
    'ttkbootstrap': 'pip install ttkbootstrap',
    'numpy': 'pip install numpy',
    'pandas': 'pip install pandas',
    'xarray': 'pip install xarray',
    'rioxarray': 'pip install rioxarray',
    'matplotlib': 'pip install matplotlib',
    'rasterio': 'pip install rasterio',
}

missing = []
for module, cmd in required.items():
    try:
        __import__(module.split('.')[0])
    except ImportError:
        missing.append(f"{module}: {cmd}")

if missing:
    print("\n❌ 缺少依赖:\n")
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
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import datetime
import threading
import os
import re
from rasterio.io import MemoryFile
from rasterio.transform import from_origin
import rasterio

# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'STHeiti']
matplotlib.rcParams['axes.unicode_minus'] = False

print("✓ 依赖加载成功\n")


# ==================== 配置 ====================
class Config:
    VERSION = "1.0"
    APP_NAME = "时序遥感分析系统"
    NODATA = -9999.0


# ==================== 工具函数 ====================
def extract_time_from_filename(filename):
    """从文件名提取时间信息"""
    # 年-月格式
    m = re.search(r'(19\d{2}|20\d{2})_?(\d{1,2})', filename)
    if m:
        year, month = int(m.group(1)), int(m.group(2))
        if 1 <= month <= 12:
            return datetime.datetime(year, month, 1)

    # 仅年份
    m = re.search(r'(19\d{2}|20\d{2})', filename)
    if m:
        return datetime.datetime(int(m.group(1)), 1, 1)

    return None


def calculate_statistics(data_array):
    """计算数据统计信息"""
    values = data_array.values
    valid = values[~np.isnan(values)]

    if len(valid) == 0:
        return None

    return {
        'min': float(np.min(valid)),
        'max': float(np.max(valid)),
        'mean': float(np.mean(valid)),
        'std': float(np.std(valid)),
        'median': float(np.median(valid)),
        'count': len(valid),
        'total': values.size
    }


def export_to_geotiff(data_array, filepath):
    """导出为GeoTIFF"""
    # 转为2D
    if hasattr(data_array, 'values'):
        arr = data_array.values
    else:
        arr = np.array(data_array)

    if arr.ndim > 2:
        arr = np.nanmean(arr, axis=0)

    arr = np.where(np.isnan(arr), Config.NODATA, arr).astype(np.float32)

    profile = {
        'driver': 'GTiff',
        'dtype': rasterio.float32,
        'count': 1,
        'height': arr.shape[0],
        'width': arr.shape[1],
        'transform': from_origin(0, arr.shape[0], 1, 1),
        'nodata': Config.NODATA,
        'compress': 'lzw'
    }

    with rasterio.open(filepath, 'w', **profile) as dst:
        dst.write(arr, 1)


# ==================== 主应用 ====================
class RSBaseApp:
    """基础应用框架"""

    def __init__(self):
        self.root = tb.Window(
            title=f"{Config.APP_NAME} V{Config.VERSION} - 基础版",
            themename="cosmo",
            size=(1500, 900)
        )

        self.data_stack = None
        self.files = []

        self._setup_ui()

        print("应用启动成功")

    def _setup_ui(self):
        """设置UI"""
        # 创建菜单
        self._create_menu()

        # 标题栏
        header = ttk.Frame(self.root)
        header.pack(fill=X, padx=10, pady=10)

        ttk.Label(
            header,
            text=f"🛰️ {Config.APP_NAME} V{Config.VERSION}",
            font=("Helvetica", 20, "bold")
        ).pack()

        ttk.Label(
            header,
            text="第一步：基础框架 - 数据加载与统计分析",
            font=("Helvetica", 11)
        ).pack(pady=5)

        # 主框架 - 左右分栏
        main_paned = ttk.PanedWindow(self.root, orient=HORIZONTAL)
        main_paned.pack(fill=BOTH, expand=True, padx=10, pady=(0, 10))

        # 左侧控制面板
        left_frame = ttk.Frame(main_paned, width=350)
        main_paned.add(left_frame, weight=1)

        # 右侧显示面板
        right_frame = ttk.Frame(main_paned)
        main_paned.add(right_frame, weight=3)

        self._setup_left_panel(left_frame)
        self._setup_right_panel(right_frame)

    def _create_menu(self):
        """创建菜单栏"""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        # 文件菜单
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="文件", menu=file_menu)
        file_menu.add_command(label="打开文件...", command=self._select_files)
        file_menu.add_separator()
        file_menu.add_command(label="退出", command=self.root.quit)

        # 帮助菜单
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="帮助", menu=help_menu)
        help_menu.add_command(label="关于", command=self._show_about)

    def _setup_left_panel(self, parent):
        """设置左侧控制面板"""
        # 文件管理区域
        file_frame = ttk.LabelFrame(parent, text="📁 数据管理", padding=15)
        file_frame.pack(fill=X, padx=10, pady=10)

        # 选择文件按钮
        ttk.Button(
            file_frame,
            text="选择 GeoTIFF 文件",
            command=self._select_files,
            bootstyle=PRIMARY,
            width=30
        ).pack(fill=X, pady=(0, 10))

        # 文件列表
        list_label = ttk.Label(file_frame, text="已选文件:", font=("Helvetica", 9, "bold"))
        list_label.pack(anchor=tk.W, pady=(0, 5))

        # 添加滚动条的文件列表
        list_container = ttk.Frame(file_frame)
        list_container.pack(fill=X, pady=(0, 10))

        scrollbar = ttk.Scrollbar(list_container)
        scrollbar.pack(side=RIGHT, fill=Y)

        self.file_listbox = tk.Listbox(
            list_container,
            height=8,
            yscrollcommand=scrollbar.set,
            font=("Consolas", 9)
        )
        self.file_listbox.pack(side=LEFT, fill=BOTH, expand=True)
        scrollbar.config(command=self.file_listbox.yview)

        # 操作按钮
        btn_frame = ttk.Frame(file_frame)
        btn_frame.pack(fill=X)

        ttk.Button(
            btn_frame,
            text="清除",
            command=self._clear_files,
            bootstyle=SECONDARY,
            width=13
        ).pack(side=LEFT, padx=(0, 5))

        ttk.Button(
            btn_frame,
            text="加载数据",
            command=self._load_data,
            bootstyle=SUCCESS,
            width=13
        ).pack(side=RIGHT)

        # 数据信息区域
        info_frame = ttk.LabelFrame(parent, text="📊 数据信息", padding=15)
        info_frame.pack(fill=X, padx=10, pady=(0, 10))

        self.info_text = tk.Text(
            info_frame,
            height=8,
            wrap=tk.WORD,
            font=("Consolas", 9),
            bg="#f8f9fa"
        )
        self.info_text.pack(fill=X)
        self.info_text.insert("1.0", "请先选择数据文件...")
        self.info_text.config(state=tk.DISABLED)

        # 快捷功能区域
        quick_frame = ttk.LabelFrame(parent, text="⚡ 快捷功能", padding=15)
        quick_frame.pack(fill=X, padx=10, pady=(0, 10))

        ttk.Button(
            quick_frame,
            text="📊 详细统计信息",
            command=self._show_detailed_stats,
            bootstyle=INFO,
            width=30
        ).pack(fill=X, pady=3)

        ttk.Button(
            quick_frame,
            text="📈 时序折线图",
            command=self._show_timeseries_plot,
            bootstyle=INFO,
            width=30
        ).pack(fill=X, pady=3)

        ttk.Button(
            quick_frame,
            text="📉 箱线图",
            command=self._show_boxplot,
            bootstyle=INFO,
            width=30
        ).pack(fill=X, pady=3)

        ttk.Button(
            quick_frame,
            text="💾 导出当前数据",
            command=self._export_data,
            bootstyle=WARNING,
            width=30
        ).pack(fill=X, pady=3)

        # 进度指示
        progress_frame = ttk.Frame(parent)
        progress_frame.pack(fill=X, padx=10, pady=(0, 10))

        self.progress_bar = ttk.Progressbar(
            progress_frame,
            mode='indeterminate',
            bootstyle="success-striped"
        )
        self.progress_bar.pack(fill=X)

        self.progress_label = ttk.Label(
            progress_frame,
            text="",
            font=("Helvetica", 9)
        )
        self.progress_label.pack(pady=5)

    def _setup_right_panel(self, parent):
        """设置右侧显示面板"""
        # 创建标签页
        self.notebook = ttk.Notebook(parent)
        self.notebook.pack(fill=BOTH, expand=True)

        # 欢迎页
        self._create_welcome_tab()

        # 数据预览页
        self.preview_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.preview_frame, text="数据预览")

        # 统计分析页
        self.stats_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.stats_frame, text="统计分析")

    def _create_welcome_tab(self):
        """创建欢迎页"""
        welcome_frame = ttk.Frame(self.notebook)
        self.notebook.add(welcome_frame, text="欢迎")

        # 创建滚动文本
        welcome_text = f"""

        🎯 欢迎使用 {Config.APP_NAME} V{Config.VERSION}

        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

        📌 当前版本：第一步 - 基础框架

        ✅ 已实现功能：

          • 多文件GeoTIFF数据加载
          • 自动时间信息提取
          • 数据预览（空间分布）
          • 时序变化可视化
          • 详细统计分析
          • 箱线图分析
          • 数据导出功能

        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

        📖 使用指南：

          1️⃣  点击左侧"选择 GeoTIFF 文件"
          2️⃣  选择多个时序影像文件
          3️⃣  点击"加载数据"读取数据
          4️⃣  查看数据预览和统计信息
          5️⃣  使用快捷功能进行分析

        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

        📁 数据要求：

          • 格式：GeoTIFF (.tif 或 .tiff)
          • 命名：文件名需包含时间信息
            - 年度数据：NDVI_2000.tif, NDVI_2001.tif
            - 月度数据：NDVI_200001.tif, NDVI_2000_01.tif
          • 一致性：所有文件空间范围必须相同

        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

        🔜 下一步将添加：

          • Theil-Sen 趋势分析
          • Mann-Kendall 显著性检验
          • 结果可视化和导出

        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

        💡 提示：

          确保基础功能正常后，再逐步添加高级分析功能
          遇到问题请查看控制台输出信息

        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        """

        text_widget = tk.Text(
            welcome_frame,
            wrap=tk.WORD,
            font=("Consolas", 10),
            bg="#f8f9fa",
            padx=30,
            pady=20
        )
        text_widget.pack(fill=BOTH, expand=True)
        text_widget.insert("1.0", welcome_text)
        text_widget.config(state=tk.DISABLED)

    # ========== 文件操作 ==========
    def _select_files(self):
        """选择文件"""
        files = filedialog.askopenfilenames(
            title="选择 GeoTIFF 文件",
            filetypes=[
                ("TIFF files", "*.tif *.tiff"),
                ("All files", "*.*")
            ]
        )

        if files:
            self.files = list(files)
            self.file_listbox.delete(0, tk.END)
            for f in self.files:
                self.file_listbox.insert(tk.END, os.path.basename(f))

            print(f"已选择 {len(self.files)} 个文件")

    def _clear_files(self):
        """清除文件列表"""
        self.files = []
        self.file_listbox.delete(0, tk.END)
        self.data_stack = None
        self._update_info("请先选择数据文件...")
        print("已清除文件列表")

    def _load_data(self):
        """加载数据"""
        if not self.files:
            messagebox.showwarning("警告", "请先选择文件")
            return

        def load_thread():
            try:
                self.progress_bar.start()
                self._update_progress("正在加载数据...")

                # 提取时间信息
                times = []
                valid_files = []

                for f in self.files:
                    t = extract_time_from_filename(os.path.basename(f))
                    if t:
                        times.append(t)
                        valid_files.append(f)
                    else:
                        print(f"警告: 无法提取时间 - {os.path.basename(f)}")

                if not valid_files:
                    self.root.after(0, lambda: messagebox.showerror(
                        "错误", "未检测到有效的时间信息"))
                    return

                print(f"有效文件: {len(valid_files)}/{len(self.files)}")

                # 按时间排序
                sorted_idx = sorted(range(len(times)), key=lambda i: times[i])
                sorted_files = [valid_files[i] for i in sorted_idx]
                sorted_times = [times[i] for i in sorted_idx]

                # 读取数据
                self._update_progress("正在读取影像...")
                data_list = []

                for i, f in enumerate(sorted_files):
                    self._update_progress(f"读取第 {i + 1}/{len(sorted_files)} 个文件...")

                    da = rxr.open_rasterio(f).squeeze()
                    if "band" in da.dims:
                        da = da.isel(band=0).drop_vars('band')
                    data_list.append(da)

                # 堆叠数据
                self._update_progress("堆叠数据...")
                stack = xr.concat(data_list, dim="time")
                stack = stack.assign_coords(time=sorted_times)
                self.data_stack = stack.transpose('time', 'y', 'x')

                print(f"数据加载成功: {self.data_stack.shape}")

                # 完成
                self.root.after(0, self._on_data_loaded)

            except Exception as e:
                print(f"加载失败: {e}")
                self.root.after(0, lambda: messagebox.showerror(
                    "错误", f"数据加载失败:\n{str(e)}"))
            finally:
                self.progress_bar.stop()
                self._update_progress("")

        threading.Thread(target=load_thread, daemon=True).start()

    def _on_data_loaded(self):
        """数据加载完成回调"""
        # 更新信息
        ny, nx = self.data_stack.sizes['y'], self.data_stack.sizes['x']
        n_time = self.data_stack.sizes['time']

        # 计算基本统计
        first_image = self.data_stack.isel(time=0)
        valid_pixels = np.sum(~np.isnan(first_image.values))
        total_pixels = first_image.size
        valid_percent = (valid_pixels / total_pixels) * 100

        info = f"""✓ 数据加载成功！

时间序列: {n_time} 期
空间大小: {ny} × {nx} 像元
数据类型: {self.data_stack.dtype}
有效像元: {valid_pixels:,} ({valid_percent:.1f}%)

时间范围:
  起始: {str(self.data_stack.time.values[0])[:10]}
  结束: {str(self.data_stack.time.values[-1])[:10]}"""

        self._update_info(info)

        # 显示预览
        self._show_data_preview()

        # 显示统计
        self._show_statistics()

        messagebox.showinfo("成功", f"数据加载完成！\n共 {n_time} 期影像")

    def _update_info(self, text):
        """更新信息显示"""
        self.info_text.config(state=tk.NORMAL)
        self.info_text.delete("1.0", tk.END)
        self.info_text.insert("1.0", text)
        self.info_text.config(state=tk.DISABLED)

    def _update_progress(self, text):
        """更新进度文本"""
        self.progress_label.config(text=text)

    # ========== 数据预览 ==========
    def _show_data_preview(self):
        """显示数据预览"""
        # 清除旧内容
        for widget in self.preview_frame.winfo_children():
            widget.destroy()

        try:
            # 创建图表
            fig, axes = plt.subplots(2, 2, figsize=(14, 12))
            fig.suptitle("数据预览", fontsize=16, fontweight='bold')

            # 第一期影像
            first_data = self.data_stack.isel(time=0).values
            im1 = axes[0, 0].imshow(first_data, cmap='viridis')
            axes[0, 0].set_title(f"第一期影像\n{str(self.data_stack.time.values[0])[:10]}",
                                 fontsize=11, fontweight='bold')
            plt.colorbar(im1, ax=axes[0, 0], fraction=0.046, pad=0.04)
            axes[0, 0].axis('off')

            # 最后一期影像
            last_data = self.data_stack.isel(time=-1).values
            im2 = axes[0, 1].imshow(last_data, cmap='viridis')
            axes[0, 1].set_title(f"最后一期影像\n{str(self.data_stack.time.values[-1])[:10]}",
                                 fontsize=11, fontweight='bold')
            plt.colorbar(im2, ax=axes[0, 1], fraction=0.046, pad=0.04)
            axes[0, 1].axis('off')

            # 时序均值
            mean_data = self.data_stack.mean(dim='time').values
            im3 = axes[1, 0].imshow(mean_data, cmap='viridis')
            axes[1, 0].set_title("时序均值", fontsize=11, fontweight='bold')
            plt.colorbar(im3, ax=axes[1, 0], fraction=0.046, pad=0.04)
            axes[1, 0].axis('off')

            # 随机像元时序
            ny, nx = self.data_stack.sizes['y'], self.data_stack.sizes['x']
            n_samples = min(8, ny * nx)

            for _ in range(n_samples):
                row = np.random.randint(0, ny)
                col = np.random.randint(0, nx)
                ts = self.data_stack[:, row, col].values

                if not np.all(np.isnan(ts)):
                    axes[1, 1].plot(range(len(ts)), ts, 'o-',
                                    markersize=4, linewidth=1.5, alpha=0.7)

            axes[1, 1].set_title("随机像元时序变化", fontsize=11, fontweight='bold')
            axes[1, 1].set_xlabel("时间索引", fontsize=10)
            axes[1, 1].set_ylabel("值", fontsize=10)
            axes[1, 1].grid(True, alpha=0.3, linestyle='--')

            plt.tight_layout()

            # 嵌入到Tkinter
            canvas = FigureCanvasTkAgg(fig, self.preview_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=BOTH, expand=True, padx=10, pady=10)

            print("数据预览已生成")

        except Exception as e:
            print(f"预览生成失败: {e}")
            error_label = ttk.Label(
                self.preview_frame,
                text=f"预览生成失败:\n{str(e)}",
                font=("Helvetica", 10)
            )
            error_label.pack(expand=True)

    # ========== 统计分析 ==========
    def _show_statistics(self):
        """显示统计分析"""
        for widget in self.stats_frame.winfo_children():
            widget.destroy()

        try:
            # 创建图表
            fig, axes = plt.subplots(2, 2, figsize=(14, 12))
            fig.suptitle("统计分析", fontsize=16, fontweight='bold')

            # 1. 空间平均时序
            mean_ts = self.data_stack.mean(dim=['y', 'x']).values
            axes[0, 0].plot(range(len(mean_ts)), mean_ts, 'o-',
                            linewidth=2, markersize=6, color='#2E86AB')
            axes[0, 0].set_title("空间平均时序", fontsize=11, fontweight='bold')
            axes[0, 0].set_xlabel("时间索引")
            axes[0, 0].set_ylabel("均值")
            axes[0, 0].grid(True, alpha=0.3)

            # 2. 标准差时序
            std_ts = self.data_stack.std(dim=['y', 'x']).values
            axes[0, 1].plot(range(len(std_ts)), std_ts, 'o-',
                            linewidth=2, markersize=6, color='#C73E1D')
            axes[0, 1].set_title("空间标准差时序", fontsize=11, fontweight='bold')
            axes[0, 1].set_xlabel("时间索引")
            axes[0, 1].set_ylabel("标准差")
            axes[0, 1].grid(True, alpha=0.3)

            # 3. 数据分布直方图（第一期）
            first_data = self.data_stack.isel(time=0).values.flatten()
            valid_data = first_data[~np.isnan(first_data)]
            axes[1, 0].hist(valid_data, bins=50, color='#F18F01', alpha=0.7, edgecolor='black')
            axes[1, 0].set_title("第一期数据分布", fontsize=11, fontweight='bold')
            axes[1, 0].set_xlabel("值")
            axes[1, 0].set_ylabel("频数")
            axes[1, 0].grid(True, alpha=0.3, axis='y')

            # 4. 所有时期的箱线图
            box_data = []
            for i in range(min(self.data_stack.sizes['time'], 20)):  # 最多显示20期
                data = self.data_stack.isel(time=i).values.flatten()
                valid = data[~np.isnan(data)]
                if len(valid) > 0:
                    box_data.append(valid)

            bp = axes[1, 1].boxplot(box_data, patch_artist=True)
            for patch in bp['boxes']:
                patch.set_facecolor('#92C5DE')
            axes[1, 1].set_title("各期数据箱线图", fontsize=11, fontweight='bold')
            axes[1, 1].set_xlabel("时间索引")
            axes[1, 1].set_ylabel("值")
            axes[1, 1].grid(True, alpha=0.3, axis='y')

            plt.tight_layout()

            # 嵌入到Tkinter
            canvas = FigureCanvasTkAgg(fig, self.stats_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=BOTH, expand=True, padx=10, pady=10)

            print("统计分析已生成")

        except Exception as e:
            print(f"统计分析失败: {e}")
            error_label = ttk.Label(
                self.stats_frame,
                text=f"统计分析失败:\n{str(e)}"
            )
            error_label.pack(expand=True)

    # ========== 快捷功能 ==========
    def _show_detailed_stats(self):
        """显示详细统计信息"""
        if self.data_stack is None:
            messagebox.showwarning("警告", "请先加载数据")
            return

        # 计算所有时期的统计
        stats_list = []
        for i, time in enumerate(self.data_stack.time.values):
            data = self.data_stack.isel(time=i)
            stats = calculate_statistics(data)
            if stats:
                stats['time'] = str(time)[:10]
                stats_list.append(stats)

        # 创建窗口
        win = tb.Toplevel(self.root)
        win.title("详细统计信息")
        win.geometry("800x600")

        # 创建表格
        columns = ("时间", "最小值", "最大值", "均值", "标准差", "中位数", "有效像元")
        tree = ttk.Treeview(win, columns=columns, show='headings', height=20)

        for col in columns:
            tree.heading(col, text=col)
            tree.column(col, width=100)

        # 添加滚动条
        scrollbar = ttk.Scrollbar(win, orient=VERTICAL, command=tree.yview)
        tree.configure(yscrollcommand=scrollbar.set)

        # 填充数据
        for stats in stats_list:
            tree.insert("", tk.END, values=(
                stats['time'],
                f"{stats['min']:.4f}",
                f"{stats['max']:.4f}",
                f"{stats['mean']:.4f}",
                f"{stats['std']:.4f}",
                f"{stats['median']:.4f}",
                f"{stats['count']:,}"
            ))

        tree.pack(side=LEFT, fill=BOTH, expand=True, padx=10, pady=10)
        scrollbar.pack(side=RIGHT, fill=Y, pady=10)

        # 导出按钮
        def export_stats():
            path = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv")]
            )
            if path:
                df = pd.DataFrame(stats_list)
                df.to_csv(path, index=False)
                messagebox.showinfo("成功", f"统计信息已导出:\n{path}")

        ttk.Button(win, text="导出为CSV", command=export_stats,
                   bootstyle=PRIMARY).pack(pady=10)

    def _show_timeseries_plot(self):
        """显示时序折线图"""
        if self.data_stack is None:
            messagebox.showwarning("警告", "请先加载数据")
            return

        win = tb.Toplevel(self.root)
        win.title("时序折线图")
        win.geometry("1000x600")

        # 计算统计量
        mean_ts = self.data_stack.mean(dim=['y', 'x']).values
        std_ts = self.data_stack.std(dim=['y', 'x']).values
        min_ts = self.data_stack.min(dim=['y', 'x']).values
        max_ts = self.data_stack.max(dim=['y', 'x']).values

        # 创建图表
        fig, ax = plt.subplots(figsize=(12, 6))

        x = range(len(mean_ts))

        # 绘制均值线
        ax.plot(x, mean_ts, 'o-', linewidth=2, markersize=6,
                color='#2E86AB', label='均值')

        # 绘制标准差区间
        ax.fill_between(x, mean_ts - std_ts, mean_ts + std_ts,
                        alpha=0.3, color='#2E86AB', label='±1标准差')

        # 绘制最小最大值
        ax.plot(x, min_ts, '--', linewidth=1, color='#C73E1D',
                alpha=0.5, label='最小值')
        ax.plot(x, max_ts, '--', linewidth=1, color='#F18F01',
                alpha=0.5, label='最大值')

        ax.set_title("时序变化分析", fontsize=14, fontweight='bold')
        ax.set_xlabel("时间索引", fontsize=11)
        ax.set_ylabel("值", fontsize=11)
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3, linestyle='--')

        plt.tight_layout()

        canvas = FigureCanvasTkAgg(fig, win)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=BOTH, expand=True, padx=10, pady=10)

        ttk.Button(win, text="关闭", command=win.destroy,
                   bootstyle=SECONDARY).pack(pady=10)

    def _show_boxplot(self):
        """显示箱线图"""
        if self.data_stack is None:
            messagebox.showwarning("警告", "请先加载数据")
            return

        win = tb.Toplevel(self.root)
        win.title("箱线图分析")
        win.geometry("1200x600")

        # 准备数据
        box_data = []
        labels = []
        for i, time in enumerate(self.data_stack.time.values):
            data = self.data_stack.isel(time=i).values.flatten()
            valid = data[~np.isnan(data)]
            if len(valid) > 0:
                box_data.append(valid)
                labels.append(str(time)[:10])

        # 创建图表
        fig, ax = plt.subplots(figsize=(14, 6))

        bp = ax.boxplot(box_data, labels=labels, patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('#92C5DE')
            patch.set_alpha(0.7)

        ax.set_title("各期数据分布箱线图", fontsize=14, fontweight='bold')
        ax.set_xlabel("时间", fontsize=11)
        ax.set_ylabel("值", fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')
        plt.xticks(rotation=45, ha='right')

        plt.tight_layout()

        canvas = FigureCanvasTkAgg(fig, win)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=BOTH, expand=True, padx=10, pady=10)

        ttk.Button(win, text="关闭", command=win.destroy,
                   bootstyle=SECONDARY).pack(pady=10)

    def _export_data(self):
        """导出当前数据"""
        if self.data_stack is None:
            messagebox.showwarning("警告", "请先加载数据")
            return

        # 选择导出内容
        dialog = tb.Toplevel(self.root)
        dialog.title("导出数据")
        dialog.geometry("350x200")
        dialog.transient(self.root)
        dialog.grab_set()

        ttk.Label(dialog, text="选择要导出的内容:",
                  font=("Helvetica", 11, "bold")).pack(pady=10)

        export_var = tk.StringVar(value="mean")

        ttk.Radiobutton(dialog, text="时序均值影像",
                        variable=export_var, value="mean").pack(anchor=tk.W, padx=20, pady=5)
        ttk.Radiobutton(dialog, text="第一期影像",
                        variable=export_var, value="first").pack(anchor=tk.W, padx=20, pady=5)
        ttk.Radiobutton(dialog, text="最后一期影像",
                        variable=export_var, value="last").pack(anchor=tk.W, padx=20, pady=5)

        def do_export():
            export_type = export_var.get()

            path = filedialog.asksaveasfilename(
                defaultextension=".tif",
                filetypes=[("TIFF files", "*.tif")]
            )

            if path:
                try:
                    if export_type == "mean":
                        data = self.data_stack.mean(dim='time')
                    elif export_type == "first":
                        data = self.data_stack.isel(time=0)
                    else:
                        data = self.data_stack.isel(time=-1)

                    export_to_geotiff(data, path)
                    messagebox.showinfo("成功", f"数据已导出:\n{path}")
                    dialog.destroy()
                except Exception as e:
                    messagebox.showerror("错误", f"导出失败:\n{str(e)}")

        ttk.Button(dialog, text="导出", command=do_export,
                   bootstyle=SUCCESS).pack(pady=10)

    def _show_about(self):
        """显示关于信息"""
        about_text = f"""{Config.APP_NAME} V{Config.VERSION}

第一步：基础框架

已实现功能：
• 数据加载与预览
• 统计分析
• 数据导出

开发者: @3S&ML

下一步将添加：
• Theil-Sen趋势分析
• Mann-Kendall检验"""

        messagebox.showinfo("关于", about_text)

    def run(self):
        """运行应用"""
        self.root.mainloop()


# ==================== 主程序入口 ====================
if __name__ == "__main__":
    print("=" * 70)
    print(f"{Config.APP_NAME} V{Config.VERSION}")
    print("第一步：基础框架 - 数据加载与统计分析")
    print("=" * 70)
    print()

    app = RSBaseApp()
    app.run()
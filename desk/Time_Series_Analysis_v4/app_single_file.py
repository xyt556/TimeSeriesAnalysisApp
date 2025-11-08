# app_single_file.py
"""
时序遥感分析系统 V3.0 - 完整单文件版
解决模块导入问题
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import warnings
import sys
import os

warnings.filterwarnings('ignore')

# ============ 检查依赖 ============
print("正在检查依赖包...")
required_packages = {
    'ttkbootstrap': 'ttkbootstrap',
    'numpy': 'numpy',
    'pandas': 'pandas',
    'xarray': 'xarray',
    'rioxarray': 'rioxarray',
    'matplotlib': 'matplotlib',
    'scipy': 'scipy',
    'statsmodels': 'statsmodels',
    'sklearn': 'scikit-learn',
    'rasterio': 'rasterio',
}

missing = []
for module, package in required_packages.items():
    try:
        __import__(module)
    except ImportError:
        missing.append(package)

if missing:
    print(f"\n❌ 缺少以下包: {', '.join(missing)}")
    print(f"\n安装命令: pip install {' '.join(missing)}")
    sys.exit(1)

print("✓ 所有依赖包已安装\n")

# ============ 导入库 ============
import ttkbootstrap as tb
from ttkbootstrap.constants import *
import numpy as np
import pandas as pd
import xarray as xr
import rioxarray as rxr
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import datetime
import re
import threading

# 设置中文字体
import matplotlib

matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'STHeiti']
matplotlib.rcParams['axes.unicode_minus'] = False


# ============ 主应用类 ============
class RemoteSensingApp:
    """时序遥感分析系统主类"""

    def __init__(self):
        self.root = tb.Window(
            title="时序遥感分析系统 V3.0 - 单文件版",
            themename="cosmo",
            size=(1400, 900)
        )

        self.data_stack = None
        self.uploaded_files = []
        self.analysis_results = {}

        self._setup_ui()

        print("应用程序已启动")

    def _setup_ui(self):
        """设置UI"""
        # 标题
        header_frame = ttk.Frame(self.root)
        header_frame.pack(fill=X, padx=10, pady=10)

        ttk.Label(
            header_frame,
            text="🛰️ 时序遥感分析系统 V3.0",
            font=("Helvetica", 18, "bold")
        ).pack()

        ttk.Label(
            header_frame,
            text="单文件版本 | 基础功能演示",
            font=("Helvetica", 10)
        ).pack(pady=(5, 0))

        # 主框架
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=BOTH, expand=True, padx=10, pady=(0, 10))

        # 分割窗口
        paned = ttk.PanedWindow(main_frame, orient=HORIZONTAL)
        paned.pack(fill=BOTH, expand=True)

        # 左侧控制面板
        left_frame = ttk.Frame(paned, width=300)
        paned.add(left_frame, weight=1)

        # 右侧显示面板
        right_frame = ttk.Frame(paned)
        paned.add(right_frame, weight=3)

        self._setup_left_panel(left_frame)
        self._setup_right_panel(right_frame)

    def _setup_left_panel(self, parent):
        """设置左侧控制面板"""
        # 文件上传
        file_frame = ttk.LabelFrame(parent, text="📁 数据上传", padding=10)
        file_frame.pack(fill=X, padx=5, pady=5)

        ttk.Button(
            file_frame,
            text="选择 GeoTIFF 文件",
            command=self._select_files,
            bootstyle=PRIMARY
        ).pack(fill=X, pady=5)

        self.file_listbox = tk.Listbox(file_frame, height=8)
        self.file_listbox.pack(fill=X, pady=5)

        btn_frame = ttk.Frame(file_frame)
        btn_frame.pack(fill=X, pady=5)

        ttk.Button(
            btn_frame,
            text="清除",
            command=self._clear_files,
            bootstyle=SECONDARY,
            width=12
        ).pack(side=LEFT, padx=2)

        ttk.Button(
            btn_frame,
            text="加载数据",
            command=self._load_data,
            bootstyle=SUCCESS,
            width=12
        ).pack(side=RIGHT, padx=2)

        # 数据信息
        info_frame = ttk.LabelFrame(parent, text="📊 数据信息", padding=10)
        info_frame.pack(fill=X, padx=5, pady=5)

        self.info_text = tk.Text(info_frame, height=6, wrap=tk.WORD, font=("Consolas", 9))
        self.info_text.pack(fill=X)
        self.info_text.insert("1.0", "请先上传数据文件...")
        self.info_text.config(state=tk.DISABLED)

        # 分析控制
        analysis_frame = ttk.LabelFrame(parent, text="🔧 基础分析", padding=10)
        analysis_frame.pack(fill=X, padx=5, pady=5)

        ttk.Button(
            analysis_frame,
            text="📊 显示数据统计",
            command=self._show_statistics,
            bootstyle=INFO
        ).pack(fill=X, pady=5)

        ttk.Button(
            analysis_frame,
            text="📈 显示时序图",
            command=self._show_timeseries,
            bootstyle=INFO
        ).pack(fill=X, pady=5)

        # 说明
        help_frame = ttk.LabelFrame(parent, text="💡 使用说明", padding=10)
        help_frame.pack(fill=BOTH, expand=True, padx=5, pady=5)

        help_text = """这是单文件简化版本。

功能:
• 加载GeoTIFF时序数据
• 查看数据统计
• 显示时序变化

如果此版本正常运行，
说明环境配置正确。

然后可以修复完整版的
模块导入问题。"""

        ttk.Label(
            help_frame,
            text=help_text,
            justify=LEFT,
            font=("Consolas", 9)
        ).pack(fill=BOTH, expand=True)

    def _setup_right_panel(self, parent):
        """设置右侧显示面板"""
        self.notebook = ttk.Notebook(parent)
        self.notebook.pack(fill=BOTH, expand=True)

        # 欢迎页
        welcome_frame = ttk.Frame(self.notebook)
        self.notebook.add(welcome_frame, text="欢迎")

        welcome_text = """

        欢迎使用时序遥感分析系统！

        📌 这是单文件版本，用于：
        1. 测试环境配置
        2. 验证基本功能
        3. 解决模块导入问题

        ⚡ 快速开始：
        1. 点击左侧"选择 GeoTIFF 文件"
        2. 选择多个时序遥感影像
        3. 点击"加载数据"
        4. 查看数据统计和时序变化

        📁 数据要求：
        • 格式: GeoTIFF (.tif, .tiff)
        • 时间信息在文件名中
        • 示例: NDVI_2000.tif, NDVI_2001.tif

        🔧 如果此版本运行正常：
        说明Python环境配置正确
        可以继续修复完整模块化版本

        """

        ttk.Label(
            welcome_frame,
            text=welcome_text,
            justify=LEFT,
            font=("Consolas", 10)
        ).pack(expand=True, pady=20, padx=20)

        # 预览页
        self.preview_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.preview_frame, text="数据预览")

    # ========== 功能方法 ==========

    def _select_files(self):
        """选择文件"""
        files = filedialog.askopenfilenames(
            title="选择 GeoTIFF 文件",
            filetypes=[("TIFF files", "*.tif *.tiff"), ("All files", "*.*")]
        )

        if files:
            self.uploaded_files = list(files)
            self.file_listbox.delete(0, tk.END)
            for f in files:
                self.file_listbox.insert(tk.END, os.path.basename(f))

    def _clear_files(self):
        """清除文件"""
        self.uploaded_files = []
        self.file_listbox.delete(0, tk.END)
        self.data_stack = None
        self._update_info("请先上传数据文件...")

    def _load_data(self):
        """加载数据"""
        if not self.uploaded_files:
            messagebox.showwarning("警告", "请先选择文件")
            return

        def load_thread():
            try:
                self.root.after(0, lambda: self._update_info("正在加载数据..."))

                # 提取时间并排序
                times = []
                valid_files = []

                for f in self.uploaded_files:
                    time_val = self._extract_time(os.path.basename(f))
                    if time_val:
                        times.append(time_val)
                        valid_files.append(f)

                if not valid_files:
                    self.root.after(0, lambda: messagebox.showerror("错误", "未检测到时间信息"))
                    return

                # 排序
                sorted_idx = sorted(range(len(times)), key=lambda i: times[i])
                sorted_files = [valid_files[i] for i in sorted_idx]
                sorted_times = [times[i] for i in sorted_idx]

                # 读取数据
                data_list = []
                for f in sorted_files:
                    da = rxr.open_rasterio(f).squeeze()
                    if "band" in da.dims:
                        da = da.isel(band=0).drop_vars('band')
                    data_list.append(da)

                # 堆叠
                stack = xr.concat(data_list, dim="time")
                stack = stack.assign_coords(time=sorted_times)
                stack = stack.transpose('time', 'y', 'x')

                self.data_stack = stack

                self.root.after(0, self._on_data_loaded)

            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("错误", f"加载失败:\n{str(e)}"))

        threading.Thread(target=load_thread, daemon=True).start()

    def _extract_time(self, filename):
        """从文件名提取时间"""
        # 年-月
        m = re.search(r'(19\d{2}|20\d{2})_?(\d{1,2})', filename)
        if m:
            year = int(m.group(1))
            month = int(m.group(2))
            if 1 <= month <= 12:
                return datetime.datetime(year, month, 1)

        # 年
        m = re.search(r'(19\d{2}|20\d{2})', filename)
        if m:
            return datetime.datetime(int(m.group(1)), 1, 1)

        return None

    def _on_data_loaded(self):
        """数据加载完成"""
        ny, nx = self.data_stack.sizes['y'], self.data_stack.sizes['x']
        n_time = self.data_stack.sizes['time']

        info = f"""数据加载成功！
时间序列: {n_time} 期
空间大小: {ny} × {nx}
数据类型: {self.data_stack.dtype}"""

        self._update_info(info)
        self._show_preview()

        messagebox.showinfo("成功", "数据加载完成！")

    def _update_info(self, text):
        """更新信息"""
        self.info_text.config(state=tk.NORMAL)
        self.info_text.delete("1.0", tk.END)
        self.info_text.insert("1.0", text)
        self.info_text.config(state=tk.DISABLED)

    def _show_preview(self):
        """显示预览"""
        for widget in self.preview_frame.winfo_children():
            widget.destroy()

        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

            # 第一期
            first = self.data_stack.isel(time=0)
            im1 = ax1.imshow(first.values, cmap='viridis')
            ax1.set_title("第一期影像")
            plt.colorbar(im1, ax=ax1)
            ax1.axis('off')

            # 均值
            mean = self.data_stack.mean(dim='time')
            im2 = ax2.imshow(mean.values, cmap='viridis')
            ax2.set_title("时序均值")
            plt.colorbar(im2, ax=ax2)
            ax2.axis('off')

            plt.tight_layout()

            canvas = FigureCanvasTkAgg(fig, self.preview_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=BOTH, expand=True, padx=10, pady=10)

        except Exception as e:
            ttk.Label(
                self.preview_frame,
                text=f"预览失败:\n{str(e)}"
            ).pack(expand=True)

    def _show_statistics(self):
        """显示统计信息"""
        if self.data_stack is None:
            messagebox.showwarning("警告", "请先加载数据")
            return

        # 计算统计
        mean_val = float(np.nanmean(self.data_stack.values))
        std_val = float(np.nanstd(self.data_stack.values))
        min_val = float(np.nanmin(self.data_stack.values))
        max_val = float(np.nanmax(self.data_stack.values))

        stats_text = f"""数据统计信息：

均值: {mean_val:.4f}
标准差: {std_val:.4f}
最小值: {min_val:.4f}
最大值: {max_val:.4f}"""

        messagebox.showinfo("数据统计", stats_text)

    def _show_timeseries(self):
        """显示时序图"""
        if self.data_stack is None:
            messagebox.showwarning("警告", "请先加载数据")
            return

        # 创建新窗口
        win = tb.Toplevel(self.root)
        win.title("时序变化")
        win.geometry("800x600")

        # 计算空间平均
        mean_ts = self.data_stack.mean(dim=['y', 'x']).values

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(range(len(mean_ts)), mean_ts, 'o-', linewidth=2, markersize=5)
        ax.set_title("时序变化（空间平均）", fontsize=14, fontweight='bold')
        ax.set_xlabel("时间索引")
        ax.set_ylabel("值")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        canvas = FigureCanvasTkAgg(fig, win)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=BOTH, expand=True, padx=10, pady=10)

        ttk.Button(win, text="关闭", command=win.destroy, bootstyle=SECONDARY).pack(pady=10)

    def run(self):
        """运行应用"""
        self.root.mainloop()


# ============ 主程序入口 ============
if __name__ == "__main__":
    print("=" * 60)
    print("时序遥感分析系统 V3.0 - 单文件版")
    print("=" * 60)
    print()

    app = RemoteSensingApp()
    app.run()
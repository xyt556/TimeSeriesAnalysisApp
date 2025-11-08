# visualization/interactive.py
"""
交互式可视化工具
"""

import tkinter as tk
from tkinter import ttk
import ttkbootstrap as tb
from ttkbootstrap.constants import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from utils import logger


class InteractiveTools:
    """交互式工具集"""

    @staticmethod
    def create_interactive_map(data_array, parent_window, data_stack=None):
        """创建交互式地图查看器

        Args:
            data_array: 显示的数据数组
            parent_window: 父窗口
            data_stack: 完整数据栈（用于时序查询）

        Returns:
            tk.Toplevel: 查看器窗口
        """
        viewer_window = tb.Toplevel(parent_window)
        viewer_window.title("交互式地图查看器")
        viewer_window.geometry("1200x800")

        # 准备显示数据
        if hasattr(data_array, 'dims') and "time" in data_array.dims:
            display_data = data_array.isel(time=0).values
            has_time = True
        else:
            display_data = data_array.values if hasattr(data_array, 'values') else data_array
            has_time = False

        # 创建主框架
        main_frame = ttk.Frame(viewer_window)
        main_frame.pack(fill=BOTH, expand=True)

        # 左侧：地图
        map_frame = ttk.Frame(main_frame)
        map_frame.pack(side=tk.LEFT, fill=BOTH, expand=True)

        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(display_data, cmap='viridis')
        ax.set_title("点击查看像元信息", fontsize=12, fontweight='bold')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.axis('off')

        # 右侧：信息面板
        info_frame = ttk.LabelFrame(main_frame, text="像元信息", padding=10)
        info_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=5, pady=5)

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
        btn_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=5)

        ttk.Button(btn_frame, text="关闭", command=viewer_window.destroy,
                   bootstyle=SECONDARY).pack(side=tk.RIGHT, padx=5)

        logger.info("Interactive map viewer opened")
        return viewer_window
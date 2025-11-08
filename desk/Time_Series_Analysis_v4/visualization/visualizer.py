# visualization/visualizer.py
"""
可视化模块
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from statsmodels.tsa.seasonal import STL

from config import Config
from utils import TimeExtractor, logger


class Visualizer:
    """可视化器"""

    @staticmethod
    def create_result_figure(data_array, title, cmap=None, vmin=None, vmax=None,
                             add_stats=True, figsize=(10, 8)):
        """创建结果图表

        Args:
            data_array: 数据数组
            title: 标题
            cmap: 颜色映射
            vmin, vmax: 颜色范围
            add_stats: 是否添加统计信息
            figsize: 图形大小

        Returns:
            matplotlib.figure.Figure
        """
        if cmap is None:
            cmap = Config.COLORMAPS['sequential']

        fig, ax = plt.subplots(figsize=figsize)

        data = Visualizer._prepare_data(data_array)

        # 自动计算vmin/vmax
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
                stats_text = (f'Min: {np.min(valid_data):.3f}\n'
                              f'Max: {np.max(valid_data):.3f}\n'
                              f'Mean: {np.mean(valid_data):.3f}')
                ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                        verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                        fontsize=9)

        ax.axis('off')
        plt.tight_layout()
        return fig

    @staticmethod
    def create_multi_panel_figure(data_arrays, titles, cmaps=None,
                                  figsize=(18, 5), ncols=None):
        """创建多面板图表

        Args:
            data_arrays: 数据数组列表
            titles: 标题列表
            cmaps: 颜色映射列表
            figsize: 图形大小
            ncols: 列数

        Returns:
            matplotlib.figure.Figure
        """
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
        """创建像元分析图表

        Args:
            stack: 数据栈
            row: 行索引
            col: 列索引
            period: STL周期

        Returns:
            matplotlib.figure.Figure
        """
        series = stack[:, row, col].values
        times = stack["time"].values
        time_labels = TimeExtractor.format_time_labels(times)

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

        # 4&5. STL分量
        ax4 = fig.add_subplot(gs[2, 0])
        ax5 = fig.add_subplot(gs[2, 1])
        Visualizer._add_stl_plots(ax4, ax5, series, time_labels, period)

        return fig

    @staticmethod
    def create_cluster_visualization(cluster_map, centers, times, n_samples=5):
        """创建聚类结果可视化

        Args:
            cluster_map: 聚类地图
            centers: 聚类中心
            times: 时间数组
            n_samples: 采样数

        Returns:
            matplotlib.figure.Figure
        """
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
        time_labels = TimeExtractor.format_time_labels(times)

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
        if hasattr(data_array, 'values'):
            if hasattr(data_array, 'dims') and "time" in data_array.dims:
                return np.nanmean(data_array.values, axis=0)
            return data_array.values
        return np.array(data_array)

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
                '变异系数': (np.std(valid_data) / np.mean(valid_data)
                             if np.mean(valid_data) != 0 else 0)
            }

            table_data = [[k, f'{v:.4f}'] for k, v in stats.items()]

            ax.axis('tight')
            ax.axis('off')
            table = ax.table(cellText=table_data, colLabels=['指标', '值'],
                             cellLoc='left', loc='center',
                             colWidths=[0.5, 0.5])
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 2)

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
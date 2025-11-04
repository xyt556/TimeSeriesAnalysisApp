# utils/visualization.py
import io
import numpy as np
import rasterio
from rasterio.io import MemoryFile
from rasterio.transform import from_origin
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap
import streamlit as st
import pandas as pd
import xarray as xr


# 自定义颜色映射
def create_custom_cmap():
    """创建自定义颜色映射"""
    colors = ['#2166ac', '#4393c3', '#92c5de', '#d1e5f0', '#f7f7f7',
              '#fddbc7', '#f4a582', '#d6604d', '#b2182b']
    return LinearSegmentedColormap.from_list('custom_rdbu', colors, N=256)


# 数据转换工具
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
        st.error(f"数据转换错误: {e}")
        return np.array(da)


def plot_map(da, title="", cmap=None, vmin=None, vmax=None,
             fig_width=8, fig_height=6):
    """
    修复版的栅格地图绘制函数 - 移除了 return_fig 参数
    """
    if cmap is None:
        cmap = create_custom_cmap()

    img = _da_to_2d(da)

    # 自动确定显示范围
    if vmin is None:
        vmin = np.nanpercentile(img, 2)
    if vmax is None:
        vmax = np.nanpercentile(img, 98)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    # 处理包含负值的数据
    if np.nanmin(img) < 0 and np.nanmax(img) > 0 and (vmin is None or vmin < 0):
        norm = TwoSlopeNorm(vcenter=0, vmin=vmin, vmax=vmax)
        im = ax.imshow(img, cmap=cmap, norm=norm, interpolation='nearest')
    else:
        im = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax,
                       interpolation='nearest')

    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label('值', rotation=270, labelpad=15)

    # 设置标题和坐标
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.axis("off")

    # 显示图形
    st.pyplot(fig)
    plt.close(fig)  # 释放内存


def plot_pixel_timeseries(stack, row=None, col=None, period=12):
    """
    修复版的像元时序分析函数 - 确保每个时间点只有一个值
    """
    if stack is None:
        st.warning("请先上传数据")
        return

    ny = int(stack.sizes["y"])
    nx = int(stack.sizes["x"])

    # 侧边栏控件
    st.sidebar.subheader("像元选择")
    if row is None:
        row = st.sidebar.slider("行 (Y)", 0, ny - 1, ny // 2,
                                help="选择像元的行坐标")
    if col is None:
        col = st.sidebar.slider("列 (X)", 0, nx - 1, nx // 2,
                                help="选择像元的列坐标")

    # 获取时序数据
    try:
        series = stack[:, row, col].values
        times = stack["time"].values

        # 格式化时间标签
        time_labels = []
        for t in times:
            if isinstance(t, np.datetime64):
                time_labels.append(np.datetime_as_string(t, unit='D'))
            else:
                time_labels.append(str(t))

        # 检查数据有效性
        if np.all(np.isnan(series)):
            st.warning(f"所选像元 ({row}, {col}) 数据全为NaN")
            return

        valid_mask = ~np.isnan(series)
        if np.sum(valid_mask) < 3:
            st.warning(f"所选像元 ({row}, {col}) 有效数据点不足")
            return

        # 检查时间唯一性
        if len(time_labels) != len(set(time_labels)):
            st.warning("⚠️ 检测到重复的时间点，时序分析可能不准确")

    except Exception as e:
        st.error(f"获取像元数据失败: {e}")
        return

    # 创建图表
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'像元 ({row}, {col}) 时序分析', fontsize=16, fontweight='bold')

    # 原始时序 - 确保每个时间点只有一个值
    axs[0, 0].plot(time_labels, series, 'o-', linewidth=2, markersize=4,
                   color='#2E86AB', label='原始数据')
    axs[0, 0].set_title("原始时序")
    axs[0, 0].set_ylabel("值")
    axs[0, 0].grid(True, alpha=0.3)
    axs[0, 0].legend()
    # 设置x轴标签旋转
    plt.setp(axs[0, 0].xaxis.get_majorticklabels(), rotation=45)

    # 趋势分析
    try:
        if np.sum(valid_mask) >= 3:
            # 使用数值索引而不是时间进行趋势计算
            x_numeric = np.arange(len(series))
            valid_x = x_numeric[valid_mask]
            valid_series = series[valid_mask]

            if len(valid_x) >= 2:  # 至少需要2个点才能拟合
                coeffs = np.polyfit(valid_x, valid_series, 1)
                trend_line = np.polyval(coeffs, x_numeric)

                axs[0, 1].plot(time_labels, series, 'o-', alpha=0.7, label='原始数据')
                axs[0, 1].plot(time_labels, trend_line, '--', linewidth=2,
                               color='#A23B72', label=f'趋势 (斜率: {coeffs[0]:.4f})')
                axs[0, 1].set_title("趋势分析")
                axs[0, 1].set_ylabel("值")
                axs[0, 1].grid(True, alpha=0.3)
                axs[0, 1].legend()
                plt.setp(axs[0, 1].xaxis.get_majorticklabels(), rotation=45)
            else:
                axs[0, 1].text(0.5, 0.5, "有效数据点不足\n无法计算趋势",
                               ha='center', va='center', transform=axs[0, 1].transAxes)
                axs[0, 1].set_title("趋势分析")
    except Exception as e:
        axs[0, 1].text(0.5, 0.5, f"趋势分析失败\n{e}",
                       ha='center', va='center', transform=axs[0, 1].transAxes)
        axs[0, 1].set_title("趋势分析")

    # STL分解
    try:
        from statsmodels.tsa.seasonal import STL

        if np.sum(valid_mask) >= max(3, period * 2):
            # 填充缺失值用于STL
            series_filled = series.copy()
            if not np.all(valid_mask):
                x_numeric = np.arange(len(series))
                series_filled = np.interp(x_numeric, x_numeric[valid_mask], series[valid_mask])

            stl_result = STL(series_filled, period=period, robust=True).fit()

            # 趋势分量
            axs[1, 0].plot(time_labels, stl_result.trend, linewidth=2,
                           color='#F18F01', label='趋势分量')
            axs[1, 0].set_title("STL趋势分量")
            axs[1, 0].set_xlabel("时间")
            axs[1, 0].set_ylabel("值")
            axs[1, 0].grid(True, alpha=0.3)
            axs[1, 0].legend()
            plt.setp(axs[1, 0].xaxis.get_majorticklabels(), rotation=45)

            # 季节分量
            axs[1, 1].plot(time_labels, stl_result.seasonal, linewidth=2,
                           color='#C73E1D', label='季节分量')
            axs[1, 1].set_title("STL季节分量")
            axs[1, 1].set_xlabel("时间")
            axs[1, 1].set_ylabel("值")
            axs[1, 1].grid(True, alpha=0.3)
            axs[1, 1].legend()
            plt.setp(axs[1, 1].xaxis.get_majorticklabels(), rotation=45)

        else:
            for i in range(1, 2):
                for j in range(2):
                    axs[i, j].text(0.5, 0.5, "数据不足\n无法进行STL分解",
                                   ha='center', va='center', transform=axs[i, j].transAxes)
                    axs[i, j].set_title("STL分析")
    except Exception as e:
        for i in range(1, 2):
            for j in range(2):
                axs[i, j].text(0.5, 0.5, f"STL分析失败\n{e}",
                               ha='center', va='center', transform=axs[i, j].transAxes)
                axs[i, j].set_title("STL分析")

    plt.tight_layout()
    st.pyplot(fig)

    # 下载功能
    col1, col2 = st.columns(2)

    with col1:
        # 下载PNG
        buf_png = io.BytesIO()
        fig.savefig(buf_png, format="png", bbox_inches="tight", dpi=150,
                    facecolor='white', edgecolor='none')
        buf_png.seek(0)
        st.download_button(
            "⬇️ 下载时序图 (PNG)",
            data=buf_png.getvalue(),
            file_name=f"pixel_{row}_{col}.png",
            mime="image/png",
            use_container_width=True
        )
        buf_png.close()

    with col2:
        # 下载CSV
        df = pd.DataFrame({
            "time": time_labels,
            "value": series,
            "is_valid": valid_mask
        })
        csv_data = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ 下载时序数据 (CSV)",
            data=csv_data,
            file_name=f"pixel_{row}_{col}.csv",
            mime="text/csv",
            use_container_width=True
        )

    plt.close(fig)  # 释放内存


def dataarray_to_bytes_tif(da, nodata=-9999.0):
    """
    改进的DataArray转GeoTIFF函数
    更好地保持空间参考信息
    """
    arr2d = _da_to_2d(da)

    # 处理NaN值
    arr2d = np.where(np.isnan(arr2d), nodata, arr2d).astype(np.float32)

    try:
        # 尝试使用rioxarray的profile
        if hasattr(da, 'rio') and hasattr(da.rio, 'crs') and da.rio.crs is not None:
            profile = da.rio.profile.copy()
            profile.update({
                'dtype': rasterio.float32,
                'count': 1,
                'compress': 'lzw',
                'nodata': nodata
            })
        else:
            # 创建默认profile
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

        # 写入内存文件
        memfile = MemoryFile()
        with memfile.open(**profile) as dst:
            dst.write(arr2d, 1)

        data = memfile.read()
        memfile.close()
        return data

    except Exception as e:
        st.error(f"生成GeoTIFF失败: {e}")
        # 返回空字节作为fallback
        return b''


def fig_to_bytes_png(fig, dpi=150):
    """将matplotlib图形转换为PNG字节"""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=dpi,
                facecolor='white', edgecolor='none')
    buf.seek(0)
    data = buf.read()
    buf.close()
    return data


# 批量下载功能
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
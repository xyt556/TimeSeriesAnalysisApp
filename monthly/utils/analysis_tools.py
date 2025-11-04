# utils/analysis_tools.py
import numpy as np
import xarray as xr
from scipy import stats, fftpack
from statsmodels.tsa.seasonal import STL
import warnings
import os
from tqdm import tqdm

warnings.filterwarnings("ignore")


# ---------------- Theil-Sen ----------------
def theil_sen_trend(stack: xr.DataArray):
    """
    Theil-Sen趋势分析
    """
    data = stack.values
    time_idx = np.arange(data.shape[0])
    ny, nx = data.shape[1], data.shape[2]
    slope = np.full((ny, nx), np.nan, dtype=np.float32)
    intercept = np.full((ny, nx), np.nan, dtype=np.float32)

    for i in range(ny):
        for j in range(nx):
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


# ---------------- Mann-Kendall ----------------
def mann_kendall_test(stack: xr.DataArray):
    """
    Mann-Kendall趋势检验
    """
    from scipy.stats import kendalltau
    data = stack.values
    ny, nx = data.shape[1], data.shape[2]
    out = np.full((ny, nx), np.nan, dtype=np.float32)
    time_idx = np.arange(data.shape[0])

    for i in range(ny):
        for j in range(nx):
            ts = data[:, i, j]
            if np.isnan(ts).all() or np.sum(~np.isnan(ts)) < 3:
                out[i, j] = 0
                continue
            try:
                # 移除NaN值
                mask = ~np.isnan(ts)
                valid_ts = ts[mask]
                valid_time = time_idx[mask]

                tau, p_value = kendalltau(valid_time, valid_ts)

                if np.isnan(p_value):
                    out[i, j] = 0
                elif p_value < 0.05:  # 显著性水平0.05
                    out[i, j] = 1 if tau > 0 else -1
                else:
                    out[i, j] = 0
            except Exception:
                out[i, j] = 0

    return xr.DataArray(out, dims=("y", "x"),
                        coords={"y": stack.y, "x": stack.x})


# ---------------- BFAST 突变检测 ----------------
def bfast_detection(stack: xr.DataArray, change_threshold=2.0):
    """
    BFAST突变检测
    """
    years = stack["time"].values
    data = stack.values
    n_time = data.shape[0]
    ny, nx = data.shape[1], data.shape[2]
    break_data = np.full((ny, nx), np.nan, dtype=np.float32)

    for i in tqdm(range(ny), desc="BFAST突变检测"):
        for j in range(nx):
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
                    # 返回第一个显著突变点对应的年份
                    break_idx = break_points[0]
                    break_data[i, j] = float(years[break_idx])

            except Exception:
                continue

    return xr.DataArray(break_data, dims=("y", "x"),
                        coords={"y": stack.y, "x": stack.x})


# ---------------- FFT 周期分析 ----------------
def fft_analysis(stack: xr.DataArray):
    """
    FFT周期分析
    """
    data = stack.values
    n = data.shape[0]
    ny, nx = data.shape[1], data.shape[2]
    amp = np.full((ny, nx), np.nan, dtype=np.float32)
    period = np.full((ny, nx), np.nan, dtype=np.float32)

    for i in range(ny):
        for j in range(nx):
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


# ---------------- STL 分解 ----------------
def stl_decompose_pixelwise(stack: xr.DataArray, period=12):
    """
    STL分解 - 修复版本
    返回二维统计量，不需要再计算均值
    """
    data = stack.values
    n, ny, nx = data.shape

    # 预分配结果数组 - 二维统计量
    trend_mean = np.full((ny, nx), np.nan, dtype=np.float32)
    seasonal_mean = np.full((ny, nx), np.nan, dtype=np.float32)
    resid_std = np.full((ny, nx), np.nan, dtype=np.float32)

    for i in tqdm(range(ny), desc="STL分解"):
        for j in range(nx):
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
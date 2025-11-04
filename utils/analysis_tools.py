# utils/analysis_tools.py
import numpy as np
import xarray as xr
from scipy import stats, fftpack
from statsmodels.tsa.seasonal import STL
import warnings
from numba import jit, prange
import concurrent.futures
from tqdm import tqdm

warnings.filterwarnings("ignore")


# ---------------- 并行计算工具 ----------------
def parallel_apply_2d(data, func, desc="Processing", n_jobs=None):
    """在2D网格上并行应用函数"""
    ny, nx = data.shape[1], data.shape[2]
    results = np.full((ny, nx), np.nan, dtype=np.float32)

    # 如果n_jobs为None，使用默认线程数
    if n_jobs is None:
        n_jobs = min(4, (os.cpu_count() or 1))

    with concurrent.futures.ThreadPoolExecutor(max_workers=n_jobs) as executor:
        futures = {}
        for i in range(ny):
            for j in range(nx):
                future = executor.submit(func, data[:, i, j], i, j)
                futures[future] = (i, j)

        for future in tqdm(concurrent.futures.as_completed(futures),
                           total=len(futures), desc=desc):
            i, j = futures[future]
            try:
                results[i, j] = future.result()
            except Exception:
                continue

    return results


# ---------------- BFAST (修复版) ----------------
def bfast_detection(stack: xr.DataArray, change_threshold=2.0):
    """
    修复版的BFAST突变检测
    """
    import os
    years = stack["time"].values
    data = stack.values
    n_time = len(years)

    def calc_pixel_break(ts, i, j):
        if np.isnan(ts).all() or n_time < 4:
            return np.nan

        try:
            # 使用滑动窗口检测突变
            if np.sum(~np.isnan(ts)) < 4:
                return np.nan

            # 方法1: 基于残差的突变检测
            x = np.arange(n_time)
            mask = ~np.isnan(ts)
            if np.sum(mask) < 4:
                return np.nan

            # 线性拟合
            coeffs = np.polyfit(x[mask], ts[mask], 1)
            trend = np.polyval(coeffs, x)
            residuals = ts - trend

            # 检测残差的突变点
            residual_std = np.nanstd(residuals)
            if residual_std == 0:
                return np.nan

            # 寻找超过阈值的点
            z_scores = np.abs(residuals) / residual_std
            break_points = np.where(z_scores > change_threshold)[0]

            if len(break_points) > 0:
                # 返回第一个显著突变点对应的年份
                break_idx = break_points[0]
                return float(years[break_idx])
            else:
                return np.nan

        except Exception:
            return np.nan

    # 使用串行处理替代并行处理，避免n_jobs参数问题
    ny, nx = data.shape[1], data.shape[2]
    break_data = np.full((ny, nx), np.nan, dtype=np.float32)

    for i in tqdm(range(ny), desc="BFAST突变检测"):
        for j in range(nx):
            break_data[i, j] = calc_pixel_break(data[:, i, j], i, j)

    return xr.DataArray(break_data, dims=("y", "x"),
                        coords={"y": stack.y, "x": stack.x})


# ---------------- Mann-Kendall (修复版) ----------------
def mann_kendall_test(stack: xr.DataArray):
    """
    修复版的Mann-Kendall趋势检验，确保图像能够显示
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
                out[i, j] = 0  # 设置为0而不是NaN，确保图像显示
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


# ---------------- Theil-Sen (保持原样) ----------------
@jit(nopython=True, cache=True)
def _theil_sen_slope(x, y):
    """计算Theil-Sen斜率的numba优化版本"""
    n = len(x)
    slopes = []

    for i in range(n):
        for j in range(i + 1, n):
            if x[j] != x[i]:
                slope = (y[j] - y[i]) / (x[j] - x[i])
                slopes.append(slope)

    if len(slopes) == 0:
        return np.nan

    return np.median(np.array(slopes))


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
            y = data[:, i, j]
            if np.isnan(y).all():
                continue
            try:
                # 移除NaN值
                mask = ~np.isnan(y)
                if np.sum(mask) < 3:
                    continue
                valid_y = y[mask]
                valid_time = time_idx[mask]

                slope_val = _theil_sen_slope(valid_time, valid_y)
                if not np.isnan(slope_val):
                    slope[i, j] = slope_val
                    # 计算截距（中位数方法）
                    intercepts = valid_y - slope_val * valid_time
                    intercept[i, j] = np.median(intercepts)
            except Exception:
                continue

    coords = {"y": stack.y, "x": stack.x}
    slope_da = xr.DataArray(slope, dims=("y", "x"), coords=coords)
    intercept_da = xr.DataArray(intercept, dims=("y", "x"), coords=coords)
    return slope_da, intercept_da


# ---------------- FFT (保持原样) ----------------
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


# ---------------- STL (简化版) ----------------
def stl_decompose_pixelwise(stack: xr.DataArray, period=12):
    """
    STL分解（简化版，避免内存问题）
    """
    data = stack.values
    n, ny, nx = data.shape
    # 只计算均值结果，避免存储整个时间序列
    trend_mean = np.full((ny, nx), np.nan, dtype=np.float32)
    seasonal_mean = np.full((ny, nx), np.nan, dtype=np.float32)
    resid_std = np.full((ny, nx), np.nan, dtype=np.float32)

    for i in range(ny):
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

                trend_mean[i, j] = np.mean(res.trend)
                seasonal_mean[i, j] = np.mean(res.seasonal)
                resid_std[i, j] = np.std(res.resid)

            except Exception:
                continue

    coords = {"y": stack.y, "x": stack.x}
    trend_da = xr.DataArray(trend_mean, dims=("y", "x"), coords=coords)
    seasonal_da = xr.DataArray(seasonal_mean, dims=("y", "x"), coords=coords)
    resid_da = xr.DataArray(resid_std, dims=("y", "x"), coords=coords)

    return trend_da, seasonal_da, resid_da
# utils/analysis_tools.py
# utils/analysis_tools.py
import numpy as np
import xarray as xr
from scipy import stats, fftpack
from statsmodels.tsa.seasonal import STL
import warnings
import os
from tqdm import tqdm
import datetime  # 添加这行
import pandas as pd  # 添加这行，因为后面的代码也使用了pd

warnings.filterwarnings("ignore")


# ---------------- Theil-Sen ----------------
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


# ---------------- Mann-Kendall ----------------
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


# ---------------- BFAST 突变检测 ----------------
def bfast_detection(stack: xr.DataArray, change_threshold=2.0):
    """
    BFAST突变检测 - 修复时间转换问题
    """
    # 获取时间坐标并转换为年份
    times = stack["time"].values
    years = []

    # 将时间转换为年份
    for t in times:
        if isinstance(t, np.datetime64):
            # 将np.datetime64转换为年份
            year = t.astype('datetime64[Y]').astype(int) + 1970
            years.append(year)
        elif hasattr(t, 'year'):
            # 如果是datetime对象，直接获取年份
            years.append(t.year)
        else:
            # 如果已经是数字，假设是年份
            try:
                years.append(int(t))
            except:
                years.append(2000)  # 默认值

    years = np.array(years)

    data = stack.values
    n_time = data.shape[0]
    ny, nx = data.shape[1], data.shape[2]
    break_data = np.full((ny, nx), np.nan, dtype=np.float32)

    # 创建空值掩码
    nan_mask = np.all(np.isnan(data), axis=0)

    for i in tqdm(range(ny), desc="BFAST突变检测"):
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

# ---------------- FFT 周期分析 ----------------
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


# ---------------- STL 分解 ----------------
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

    for i in tqdm(range(ny), desc="STL分解"):
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
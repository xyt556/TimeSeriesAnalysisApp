# utils/analysis_tools.py
import numpy as np
import xarray as xr
from scipy import stats, fftpack
from statsmodels.tsa.seasonal import STL
import warnings
warnings.filterwarnings("ignore")

# ---------------- Theil-Sen ----------------
def theil_sen_trend(stack: xr.DataArray):
    """
    输入 stack: xarray.DataArray with dims ("time","y","x")
    返回 slope_da, intercept_da (DataArray dims "y","x")
    """
    data = stack.values  # shape (time, y, x)
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
                res = stats.theilslopes(y, time_idx)
                # res can be 4 or 5 items, but first two are slope and intercept
                slope[i, j] = float(res[0])
                intercept[i, j] = float(res[1])
            except Exception:
                continue

    coords = {"y": stack.y, "x": stack.x}
    slope_da = xr.DataArray(slope, dims=("y", "x"), coords=coords)
    intercept_da = xr.DataArray(intercept, dims=("y", "x"), coords=coords)
    return slope_da, intercept_da

# ---------------- Mann-Kendall ----------------
def mann_kendall_test(stack: xr.DataArray):
    """
    返回 DataArray (y,x)： 1 = 显著上升 (p<0.05), -1 = 显著下降, 0 = 无显著
    """
    from scipy.stats import kendalltau
    data = stack.values
    ny, nx = data.shape[1], data.shape[2]
    out = np.full((ny, nx), np.nan, dtype=np.float32)
    time_idx = np.arange(data.shape[0])

    for i in range(ny):
        for j in range(nx):
            y = data[:, i, j]
            if np.isnan(y).all():
                continue
            try:
                tau, p = kendalltau(time_idx, y)
                if np.isnan(p):
                    out[i, j] = 0
                elif p < 0.05:
                    out[i, j] = 1 if tau > 0 else -1
                else:
                    out[i, j] = 0
            except Exception:
                out[i, j] = 0

    return xr.DataArray(out, dims=("y", "x"), coords={"y": stack.y, "x": stack.x})

# ---------------- BFAST (简化差分法) ----------------
def bfast_detection(stack: xr.DataArray):
    """
    简化版：基于一阶差分的阈值查找突变点，返回突变年份或 NaN
    """
    years = stack["time"].values
    data = stack.values
    ny, nx = data.shape[1], data.shape[2]
    out = np.full((ny, nx), np.nan, dtype=np.float32)

    for i in range(ny):
        for j in range(nx):
            ts = data[:, i, j]
            if np.isnan(ts).all():
                continue
            diff = np.diff(ts)
            if len(diff) < 2:
                continue
            mu, sigma = np.nanmean(diff), np.nanstd(diff)
            if np.isnan(sigma) or sigma == 0:
                continue
            idx = np.where(np.abs(diff - mu) > 2 * sigma)[0]
            if idx.size > 0:
                out[i, j] = float(years[idx[0] + 1])
    return xr.DataArray(out, dims=("y", "x"), coords={"y": stack.y, "x": stack.x})

# ---------------- FFT ----------------
def fft_analysis(stack: xr.DataArray):
    """
    返回 (amplitude_da, period_da)
    amplitude: 主频振幅（float）
    period: 主频对应的周期（时间单位，与 time 索引步长一致）
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
            y = ts - np.nanmean(ts)
            yf = fftpack.fft(y)
            xf = fftpack.fftfreq(n, d=1)
            half = n // 2
            power = np.abs(yf[:half])
            power[0] = 0
            if power.size <= 1:
                continue
            idx = int(np.nanargmax(power[1:])) + 1 if power.size > 1 else 0
            amp[i, j] = float(power[idx])
            freq = xf[idx] if xf[idx] != 0 else np.nan
            period[i, j] = float(1.0 / freq) if (not np.isnan(freq) and freq != 0) else np.nan

    return xr.DataArray(amp, dims=("y", "x"), coords={"y": stack.y, "x": stack.x}), \
           xr.DataArray(period, dims=("y", "x"), coords={"y": stack.y, "x": stack.x})

# ---------------- STL ----------------
def stl_decompose_pixelwise(stack: xr.DataArray, period=12):
    """
    逐像元 STL 分解，返回 trend_da, seasonal_da, resid_da (dims: time,y,x)
    """
    data = stack.values
    n, ny, nx = data.shape
    trend = np.full_like(data, np.nan, dtype=np.float32)
    seasonal = np.full_like(data, np.nan, dtype=np.float32)
    resid = np.full_like(data, np.nan, dtype=np.float32)

    for i in range(ny):
        for j in range(nx):
            ts = data[:, i, j]
            if np.isnan(ts).all():
                continue
            try:
                stl = STL(ts, period=period, robust=True)
                res = stl.fit()
                trend[:, i, j] = res.trend
                seasonal[:, i, j] = res.seasonal
                resid[:, i, j] = res.resid
            except Exception:
                continue

    coords = stack.coords
    trend_da = xr.DataArray(trend, dims=("time", "y", "x"), coords=coords)
    seasonal_da = xr.DataArray(seasonal, dims=("time", "y", "x"), coords=coords)
    resid_da = xr.DataArray(resid, dims=("time", "y", "x"), coords=coords)
    return trend_da, seasonal_da, resid_da

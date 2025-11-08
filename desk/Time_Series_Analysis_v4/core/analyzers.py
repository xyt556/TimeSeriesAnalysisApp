# core/analyzers.py
"""
核心分析算法模块
包含趋势分析、突变检测、频率分析、STL分解等
"""

import numpy as np
import xarray as xr
from scipy import stats, fftpack
from statsmodels.tsa.seasonal import STL
import pandas as pd

from config import Config
from utils import logger, TimeExtractor


class TrendAnalyzer:
    """趋势分析器"""

    @staticmethod
    def theil_sen(stack: xr.DataArray, progress_tracker=None):
        """Theil-Sen趋势分析

        Args:
            stack: 时间序列数据
            progress_tracker: 进度跟踪器

        Returns:
            (slope, intercept): 斜率和截距的DataArray
        """
        logger.info("Starting Theil-Sen trend analysis")

        data = stack.values
        time_idx = np.arange(data.shape[0])
        ny, nx = data.shape[1], data.shape[2]

        slope = np.full((ny, nx), np.nan, dtype=np.float32)
        intercept = np.full((ny, nx), np.nan, dtype=np.float32)
        nan_mask = np.all(np.isnan(data), axis=0)

        total_pixels = ny * nx
        processed = 0

        for i in range(ny):
            if progress_tracker and progress_tracker.is_cancelled:
                break

            for j in range(nx):
                if nan_mask[i, j]:
                    continue

                ts = data[:, i, j]
                if not np.isnan(ts).all():
                    try:
                        res = stats.theilslopes(ts, time_idx)
                        slope[i, j] = res[0]
                        intercept[i, j] = res[1]
                    except:
                        continue

                processed += 1
                if progress_tracker and processed % 1000 == 0:
                    progress_tracker.update(
                        "Theil-Sen分析中",
                        (processed / total_pixels) * 100
                    )

        coords = {"y": stack.y, "x": stack.x}
        logger.info(f"Theil-Sen analysis completed: {processed} pixels processed")

        return (
            xr.DataArray(slope, dims=("y", "x"), coords=coords),
            xr.DataArray(intercept, dims=("y", "x"), coords=coords)
        )

    @staticmethod
    def mann_kendall(stack: xr.DataArray, significance=None,
                     progress_tracker=None):
        """Mann-Kendall趋势检验

        Args:
            stack: 时间序列数据
            significance: 显著性水平
            progress_tracker: 进度跟踪器

        Returns:
            DataArray: 趋势检验结果 (1=显著上升, -1=显著下降, 0=无显著趋势)
        """
        if significance is None:
            significance = Config.MK_SIGNIFICANCE

        logger.info(f"Starting Mann-Kendall test (significance={significance})")

        from scipy.stats import kendalltau

        data = stack.values
        ny, nx = data.shape[1], data.shape[2]
        out = np.full((ny, nx), np.nan, dtype=np.float32)
        time_idx = np.arange(data.shape[0])
        nan_mask = np.all(np.isnan(data), axis=0)

        total_pixels = ny * nx
        processed = 0

        for i in range(ny):
            if progress_tracker and progress_tracker.is_cancelled:
                break

            for j in range(nx):
                if nan_mask[i, j]:
                    continue

                ts = data[:, i, j]
                mask = ~np.isnan(ts)

                if np.sum(mask) < 3:
                    continue

                try:
                    valid_ts = ts[mask]
                    valid_time = time_idx[mask]
                    tau, p_value = kendalltau(valid_time, valid_ts)

                    if not np.isnan(p_value) and not np.isnan(tau):
                        if p_value < significance:
                            out[i, j] = 1.0 if tau > 0 else -1.0
                        else:
                            out[i, j] = 0.0
                except:
                    continue

                processed += 1
                if progress_tracker and processed % 1000 == 0:
                    progress_tracker.update(
                        "Mann-Kendall检验中",
                        (processed / total_pixels) * 100
                    )

        logger.info(f"Mann-Kendall test completed: {processed} pixels processed")
        return xr.DataArray(out, dims=("y", "x"),
                            coords={"y": stack.y, "x": stack.x})


class BreakpointDetector:
    """突变点检测器"""

    @staticmethod
    def bfast(stack: xr.DataArray, threshold=None, progress_tracker=None):
        """BFAST突变检测

        Args:
            stack: 时间序列数据
            threshold: 阈值
            progress_tracker: 进度跟踪器

        Returns:
            DataArray: 突变年份
        """
        if threshold is None:
            threshold = Config.BFAST_THRESHOLD

        logger.info(f"Starting BFAST detection (threshold={threshold})")

        times = stack["time"].values
        years = TimeExtractor.convert_to_years(times)

        data = stack.values
        n_time = data.shape[0]
        ny, nx = data.shape[1], data.shape[2]
        break_data = np.full((ny, nx), np.nan, dtype=np.float32)
        nan_mask = np.all(np.isnan(data), axis=0)

        total_pixels = ny * nx
        processed = 0

        for i in range(ny):
            if progress_tracker and progress_tracker.is_cancelled:
                break

            for j in range(nx):
                if nan_mask[i, j]:
                    continue

                ts = data[:, i, j]
                mask = ~np.isnan(ts)

                if np.sum(mask) < 4:
                    continue

                try:
                    x = np.arange(n_time)
                    coeffs = np.polyfit(x[mask], ts[mask], 1)
                    trend = np.polyval(coeffs, x)
                    residuals = ts - trend
                    residual_std = np.nanstd(residuals)

                    if residual_std > 0:
                        z_scores = np.abs(residuals) / residual_std
                        break_points = np.where(z_scores > threshold)[0]

                        if len(break_points) > 0:
                            break_idx = break_points[0]
                            break_data[i, j] = float(years[break_idx])
                except:
                    continue

                processed += 1
                if progress_tracker and processed % 1000 == 0:
                    progress_tracker.update(
                        "BFAST突变检测中",
                        (processed / total_pixels) * 100
                    )

        result = xr.DataArray(break_data, dims=("y", "x"),
                              coords={"y": stack.y, "x": stack.x})
        logger.info(f"BFAST detection completed: {processed} pixels processed")

        return BreakpointDetector._fix_results(result)

    @staticmethod
    def _fix_results(break_da):
        """修复BFAST结果，确保年份合理"""
        import datetime

        break_values = break_da.values
        break_values_fixed = np.full_like(break_values, np.nan)
        current_year = datetime.datetime.now().year

        for i in range(break_values.shape[0]):
            for j in range(break_values.shape[1]):
                val = break_values[i, j]
                if not np.isnan(val):
                    if val > 1e18:  # 时间戳格式
                        try:
                            dt = pd.to_datetime(val)
                            if 1900 <= dt.year <= current_year + 1:
                                break_values_fixed[i, j] = dt.year
                        except:
                            pass
                    elif 1900 <= val <= current_year + 1:
                        break_values_fixed[i, j] = val

        return xr.DataArray(break_values_fixed, dims=break_da.dims,
                            coords=break_da.coords)


class FrequencyAnalyzer:
    """频率分析器"""

    @staticmethod
    def fft(stack: xr.DataArray, progress_tracker=None):
        """FFT周期分析

        Args:
            stack: 时间序列数据
            progress_tracker: 进度跟踪器

        Returns:
            (amplitude, period): 振幅和周期的DataArray
        """
        logger.info("Starting FFT analysis")

        data = stack.values
        n = data.shape[0]
        ny, nx = data.shape[1], data.shape[2]

        amp = np.full((ny, nx), np.nan, dtype=np.float32)
        period = np.full((ny, nx), np.nan, dtype=np.float32)
        nan_mask = np.all(np.isnan(data), axis=0)

        total_pixels = ny * nx
        processed = 0

        for i in range(ny):
            if progress_tracker and progress_tracker.is_cancelled:
                break

            for j in range(nx):
                if nan_mask[i, j]:
                    continue

                ts = data[:, i, j]
                if np.isnan(ts).all():
                    continue

                try:
                    y = ts - np.nanmean(ts)
                    yf = fftpack.fft(y)
                    xf = fftpack.fftfreq(n, d=1)

                    half = n // 2
                    power = np.abs(yf[:half])
                    power[0] = 0  # 忽略直流分量

                    if power.size > 1:
                        idx = np.argmax(power[1:]) + 1
                        amp[i, j] = float(power[idx])

                        freq = xf[idx]
                        if freq != 0:
                            period[i, j] = float(1.0 / freq)
                except:
                    continue

                processed += 1
                if progress_tracker and processed % 1000 == 0:
                    progress_tracker.update(
                        "FFT分析中",
                        (processed / total_pixels) * 100
                    )

        coords = {"y": stack.y, "x": stack.x}
        logger.info(f"FFT analysis completed: {processed} pixels processed")

        return (
            xr.DataArray(amp, dims=("y", "x"), coords=coords),
            xr.DataArray(period, dims=("y", "x"), coords=coords)
        )


class STLDecomposer:
    """STL分解器"""

    @staticmethod
    def decompose(stack: xr.DataArray, period=None, progress_tracker=None):
        """STL时间序列分解

        Args:
            stack: 时间序列数据
            period: 周期
            progress_tracker: 进度跟踪器

        Returns:
            (trend, seasonal, resid): 趋势、季节、残差的DataArray
        """
        if period is None:
            period = Config.STL_DEFAULT_PERIOD

        logger.info(f"Starting STL decomposition (period={period})")

        data = stack.values
        n, ny, nx = data.shape

        trend_mean = np.full((ny, nx), np.nan, dtype=np.float32)
        seasonal_mean = np.full((ny, nx), np.nan, dtype=np.float32)
        resid_std = np.full((ny, nx), np.nan, dtype=np.float32)
        nan_mask = np.all(np.isnan(data), axis=0)

        total_pixels = ny * nx
        processed = 0

        for i in range(ny):
            if progress_tracker and progress_tracker.is_cancelled:
                break

            for j in range(nx):
                if nan_mask[i, j]:
                    continue

                ts = data[:, i, j]
                if np.sum(~np.isnan(ts)) < period * 2:
                    continue

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
                    continue

                processed += 1
                if progress_tracker and processed % 1000 == 0:
                    progress_tracker.update(
                        "STL分解中",
                        (processed / total_pixels) * 100
                    )

        coords = {"y": stack.y, "x": stack.x}
        logger.info(f"STL decomposition completed: {processed} pixels processed")

        return (
            xr.DataArray(trend_mean, dims=("y", "x"), coords=coords),
            xr.DataArray(seasonal_mean, dims=("y", "x"), coords=coords),
            xr.DataArray(resid_std, dims=("y", "x"), coords=coords)
        )
# core/preprocessors.py
"""
数据预处理模块
包含平滑、异常值检测、插值等功能
"""

import numpy as np
import xarray as xr
from scipy.signal import savgol_filter
from scipy.interpolate import CubicSpline

from config import Config
from utils import logger


class DataPreprocessor:
    """数据预处理器"""

    @staticmethod
    def smooth_savgol(stack: xr.DataArray, window_length=None, polyorder=None,
                      progress_tracker=None):
        """Savitzky-Golay平滑滤波

        Args:
            stack: 时间序列数据
            window_length: 窗口长度（必须为奇数）
            polyorder: 多项式阶数
            progress_tracker: 进度跟踪器

        Returns:
            平滑后的DataArray
        """
        if window_length is None:
            window_length = Config.SMOOTH_WINDOW
        if polyorder is None:
            polyorder = Config.SMOOTH_POLYORDER

        logger.info(f"Starting Savitzky-Golay smoothing (window={window_length}, poly={polyorder})")

        data = stack.values
        n_time, ny, nx = data.shape
        smoothed = np.full_like(data, np.nan)

        total = ny * nx
        processed = 0

        for i in range(ny):
            if progress_tracker and progress_tracker.is_cancelled:
                break
            for j in range(nx):
                ts = data[:, i, j]
                mask = ~np.isnan(ts)

                if np.sum(mask) >= window_length:
                    try:
                        valid_indices = np.where(mask)[0]
                        valid_ts = ts[mask]

                        if len(valid_ts) >= window_length:
                            smoothed_ts = savgol_filter(valid_ts, window_length, polyorder)
                            smoothed[valid_indices, i, j] = smoothed_ts
                        else:
                            smoothed[:, i, j] = ts
                    except Exception as e:
                        smoothed[:, i, j] = ts
                else:
                    smoothed[:, i, j] = ts

                processed += 1
                if progress_tracker and processed % 1000 == 0:
                    progress_tracker.update("数据平滑中", (processed / total) * 100)

        result = stack.copy(deep=True)
        result.values = smoothed
        logger.info(f"Smoothing completed: {processed} pixels processed")
        return result

    @staticmethod
    def smooth_moving_average(stack: xr.DataArray, window=3, progress_tracker=None):
        """移动平均平滑

        Args:
            stack: 时间序列数据
            window: 窗口大小
            progress_tracker: 进度跟踪器

        Returns:
            平滑后的DataArray
        """
        logger.info(f"Starting moving average smoothing (window={window})")

        data = stack.values
        n_time, ny, nx = data.shape
        smoothed = np.full_like(data, np.nan)

        total = ny * nx
        processed = 0

        for i in range(ny):
            if progress_tracker and progress_tracker.is_cancelled:
                break
            for j in range(nx):
                ts = data[:, i, j]
                mask = ~np.isnan(ts)

                if np.sum(mask) >= window:
                    for t in range(n_time):
                        start = max(0, t - window // 2)
                        end = min(n_time, t + window // 2 + 1)
                        window_data = ts[start:end]
                        smoothed[t, i, j] = np.nanmean(window_data)
                else:
                    smoothed[:, i, j] = ts

                processed += 1
                if progress_tracker and processed % 1000 == 0:
                    progress_tracker.update("移动平均中", (processed / total) * 100)

        result = stack.copy(deep=True)
        result.values = smoothed
        logger.info(f"Moving average completed: {processed} pixels processed")
        return result

    @staticmethod
    def detect_outliers(stack: xr.DataArray, method='zscore', threshold=None,
                        progress_tracker=None):
        """异常值检测

        Args:
            stack: 时间序列数据
            method: 检测方法 ('zscore' 或 'iqr')
            threshold: 阈值
            progress_tracker: 进度跟踪器

        Returns:
            布尔数组，True表示异常值
        """
        if threshold is None:
            threshold = Config.OUTLIER_THRESHOLD

        logger.info(f"Starting outlier detection (method={method}, threshold={threshold})")

        data = stack.values
        n_time, ny, nx = data.shape
        outlier_mask = np.zeros_like(data, dtype=bool)

        total = ny * nx
        processed = 0
        outlier_count = 0

        for i in range(ny):
            if progress_tracker and progress_tracker.is_cancelled:
                break
            for j in range(nx):
                ts = data[:, i, j]
                mask = ~np.isnan(ts)

                if np.sum(mask) >= 3:
                    valid_ts = ts[mask]
                    valid_indices = np.where(mask)[0]

                    if method == 'zscore':
                        mean = np.mean(valid_ts)
                        std = np.std(valid_ts)
                        if std > 0:
                            z_scores = np.abs((valid_ts - mean) / std)
                            outliers = z_scores > threshold
                        else:
                            outliers = np.zeros(len(valid_ts), dtype=bool)

                    elif method == 'iqr':
                        q1, q3 = np.percentile(valid_ts, [25, 75])
                        iqr = q3 - q1
                        lower = q1 - threshold * iqr
                        upper = q3 + threshold * iqr
                        outliers = (valid_ts < lower) | (valid_ts > upper)

                    outlier_mask[valid_indices, i, j] = outliers
                    outlier_count += np.sum(outliers)

                processed += 1
                if progress_tracker and processed % 1000 == 0:
                    progress_tracker.update("异常值检测中", (processed / total) * 100)

        logger.info(f"Outlier detection completed: {outlier_count} outliers found")
        return outlier_mask

    @staticmethod
    def remove_outliers(stack: xr.DataArray, outlier_mask, replace_method='interpolate'):
        """移除异常值

        Args:
            stack: 时间序列数据
            outlier_mask: 异常值掩码
            replace_method: 替换方法 ('nan', 'interpolate', 'mean')

        Returns:
            处理后的DataArray
        """
        logger.info(f"Removing outliers (method={replace_method})")

        data = stack.values.copy()
        n_time, ny, nx = data.shape

        if replace_method == 'nan':
            data[outlier_mask] = np.nan

        elif replace_method == 'interpolate':
            for i in range(ny):
                for j in range(nx):
                    ts = data[:, i, j]
                    outliers = outlier_mask[:, i, j]

                    if np.any(outliers):
                        ts[outliers] = np.nan
                        mask = ~np.isnan(ts)

                        if np.sum(mask) >= 2:
                            x = np.arange(n_time)
                            ts_interp = np.interp(x, x[mask], ts[mask])
                            data[:, i, j] = ts_interp

        elif replace_method == 'mean':
            for i in range(ny):
                for j in range(nx):
                    ts = data[:, i, j]
                    outliers = outlier_mask[:, i, j]

                    if np.any(outliers):
                        valid_mean = np.nanmean(ts[~outliers])
                        data[outliers, i, j] = valid_mean

        result = stack.copy(deep=True)
        result.values = data
        logger.info("Outlier removal completed")
        return result

    @staticmethod
    def interpolate_gaps(stack: xr.DataArray, method='linear', progress_tracker=None):
        """插值填补缺失值

        Args:
            stack: 时间序列数据
            method: 插值方法 ('linear', 'cubic', 'nearest')
            progress_tracker: 进度跟踪器

        Returns:
            插值后的DataArray
        """
        logger.info(f"Starting gap interpolation (method={method})")

        data = stack.values
        n_time, ny, nx = data.shape
        interpolated = data.copy()

        total = ny * nx
        processed = 0

        for i in range(ny):
            if progress_tracker and progress_tracker.is_cancelled:
                break
            for j in range(nx):
                ts = data[:, i, j]
                mask = ~np.isnan(ts)

                if np.sum(mask) >= 2 and np.sum(~mask) > 0:
                    x = np.arange(n_time)

                    try:
                        if method == 'linear':
                            interpolated[:, i, j] = np.interp(x, x[mask], ts[mask])
                        elif method == 'cubic' and np.sum(mask) >= 4:
                            cs = CubicSpline(x[mask], ts[mask])
                            interpolated[:, i, j] = cs(x)
                        elif method == 'nearest':
                            for t in range(n_time):
                                if not mask[t]:
                                    valid_indices = x[mask]
                                    nearest_idx = valid_indices[np.argmin(np.abs(valid_indices - t))]
                                    interpolated[t, i, j] = ts[nearest_idx]
                    except Exception as e:
                        logger.warning(f"Interpolation failed at ({i},{j}): {e}")
                        continue

                processed += 1
                if progress_tracker and processed % 1000 == 0:
                    progress_tracker.update("数据插值中", (processed / total) * 100)

        result = stack.copy(deep=True)
        result.values = interpolated
        logger.info(f"Interpolation completed: {processed} pixels processed")
        return result

    @staticmethod
    def spatial_subset(stack: xr.DataArray, bbox):
        """空间裁剪

        Args:
            stack: 数据
            bbox: 边界框 (row_min, row_max, col_min, col_max)

        Returns:
            裁剪后的DataArray
        """
        try:
            row_min, row_max, col_min, col_max = bbox
            subset = stack.isel(y=slice(row_min, row_max), x=slice(col_min, col_max))
            logger.info(f"Spatial subset created: shape {subset.shape}")
            return subset
        except Exception as e:
            logger.error(f"Spatial subset failed: {e}")
            return stack

    @staticmethod
    def temporal_subset(stack: xr.DataArray, start_date, end_date):
        """时间子集提取

        Args:
            stack: 数据
            start_date: 起始日期
            end_date: 结束日期

        Returns:
            时间子集DataArray
        """
        try:
            subset = stack.sel(time=slice(start_date, end_date))
            logger.info(f"Temporal subset created: {len(subset.time)} time steps")
            return subset
        except Exception as e:
            logger.error(f"Temporal subset failed: {e}")
            return stack
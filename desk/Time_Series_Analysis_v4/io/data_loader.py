# io/data_loader.py
"""
数据加载模块
"""

import os
import xarray as xr
import rioxarray as rxr

from config import Config
from utils import logger, TimeExtractor


class DataLoader:
    """数据加载器"""

    @staticmethod
    def load_timeseries_stack(file_paths, progress_callback=None):
        """加载时间序列数据栈

        Args:
            file_paths: 文件路径列表
            progress_callback: 进度回调函数

        Returns:
            xarray.DataArray: 时间序列数据栈
        """
        logger.info(f"Loading {len(file_paths)} files")

        # 提取时间信息
        times = []
        valid_files = []

        for file_path in file_paths:
            filename = os.path.basename(file_path)
            time_val = TimeExtractor.extract_time(filename)
            if time_val is not None:
                times.append(time_val)
                valid_files.append(file_path)
            else:
                logger.warning(f"Cannot extract time from: {filename}")

        if not valid_files:
            raise ValueError("未检测到有效的时间信息")

        # 按时间排序
        sorted_indices = sorted(range(len(times)), key=lambda i: times[i])
        sorted_files = [valid_files[i] for i in sorted_indices]
        sorted_times = [times[i] for i in sorted_indices]

        # 读取数据
        data_list = []
        for i, file_path in enumerate(sorted_files):
            try:
                da = rxr.open_rasterio(
                    file_path,
                    chunks=Config.CHUNK_SIZE
                ).squeeze()

                if "band" in da.dims:
                    da = da.isel(band=0).drop_vars('band')

                data_list.append(da)

                if progress_callback:
                    progress = ((i + 1) / len(sorted_files)) * 100
                    progress_callback("读取文件中", progress)

            except Exception as e:
                logger.error(f"Failed to read {file_path}: {e}")

        if not data_list:
            raise ValueError("没有成功读取任何文件")

        # 堆叠数据
        stack = xr.concat(data_list, dim="time")
        stack = stack.assign_coords(time=sorted_times)
        stack = stack.transpose('time', 'y', 'x')

        logger.info(f"Data stack created: shape {stack.shape}")
        return stack

    @staticmethod
    def get_data_info(stack):
        """获取数据基本信息

        Args:
            stack: 数据栈

        Returns:
            dict: 数据信息字典
        """
        times = stack.time.values

        info = {
            'n_time': len(times),
            'ny': stack.sizes['y'],
            'nx': stack.sizes['x'],
            'data_type': str(stack.dtype),
            'frequency': TimeExtractor.detect_frequency(times),
            'time_range': TimeExtractor.format_time_range(times)
        }

        # 计算有效像元
        sample_data = stack.isel(time=0).values
        valid_pixels = int(np.sum(~np.isnan(sample_data)))
        total_pixels = sample_data.size

        info['valid_pixels'] = valid_pixels
        info['total_pixels'] = total_pixels
        info['valid_percentage'] = (valid_pixels / total_pixels) * 100

        return info
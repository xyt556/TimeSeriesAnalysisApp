# io/exporter.py
"""
数据导出模块
"""

import numpy as np
import xarray as xr
import rasterio
from rasterio.io import MemoryFile
from rasterio.transform import from_origin

from config import Config
from utils import logger


class DataExporter:
    """数据导出器"""

    @staticmethod
    def to_geotiff(data_array, reference_stack=None, nodata=None):
        """转换为GeoTIFF字节数据

        Args:
            data_array: 数据数组
            reference_stack: 参考数据栈（用于获取CRS）
            nodata: 空值

        Returns:
            bytes: GeoTIFF字节数据
        """
        if nodata is None:
            nodata = Config.NODATA_VALUE

        # 转换为2D数组
        arr2d = DataExporter._to_2d_array(data_array)
        arr2d = np.where(np.isnan(arr2d), nodata, arr2d).astype(np.float32)

        try:
            # 获取空间参考信息
            crs, transform = DataExporter._get_spatial_reference(
                data_array, reference_stack
            )

            # 创建配置
            profile = {
                'driver': 'GTiff',
                'dtype': rasterio.float32,
                'count': 1,
                'height': arr2d.shape[0],
                'width': arr2d.shape[1],
                'compress': 'lzw',
                'nodata': nodata
            }

            if crs is not None:
                profile['crs'] = crs
            if transform is not None:
                profile['transform'] = transform
            else:
                profile['transform'] = from_origin(0, arr2d.shape[0], 1, 1)

            # 写入内存文件
            memfile = MemoryFile()
            with memfile.open(**profile) as dst:
                dst.write(arr2d, 1)

            data = memfile.read()
            memfile.close()
            return data

        except Exception as e:
            logger.error(f"GeoTIFF generation failed: {e}")
            return DataExporter._create_simple_tiff(arr2d, nodata)

    @staticmethod
    def to_csv(data_array, output_path, include_coords=True):
        """导出为CSV格式

        Args:
            data_array: 数据数组
            output_path: 输出路径
            include_coords: 是否包含坐标

        Returns:
            str: 输出文件路径
        """
        try:
            df = data_array.to_dataframe(name='value')
            if not include_coords:
                df = df.reset_index(drop=True)
            df.to_csv(output_path, index=include_coords)
            logger.info(f"CSV exported: {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"CSV export failed: {e}")
            raise

    @staticmethod
    def _to_2d_array(da):
        """转换为2D数组"""
        if isinstance(da, xr.DataArray):
            if "time" in da.dims and "y" in da.dims and "x" in da.dims:
                return np.nanmean(da.values, axis=0)
            elif "y" in da.dims and "x" in da.dims:
                return da.values
            else:
                vals = da.values
                if vals.ndim >= 2:
                    return np.nanmean(vals, axis=tuple(range(vals.ndim - 2)))
                return vals
        return np.array(da)

    @staticmethod
    def _get_spatial_reference(data_array, reference_stack):
        """获取空间参考信息"""
        crs = None
        transform = None

        if hasattr(data_array, 'rio') and data_array.rio.crs is not None:
            crs = data_array.rio.crs
            transform = data_array.rio.transform()

        if crs is None and reference_stack is not None:
            try:
                ref_da = (reference_stack.isel(time=0)
                          if 'time' in reference_stack.dims
                          else reference_stack)
                if hasattr(ref_da, 'rio') and ref_da.rio.crs is not None:
                    crs = ref_da.rio.crs
                    transform = ref_da.rio.transform()
            except:
                pass

        if transform is None:
            transform = DataExporter._infer_transform(data_array)

        return crs, transform

    @staticmethod
    def _infer_transform(da):
        """从坐标推断变换"""
        try:
            if hasattr(da, 'x') and hasattr(da, 'y'):
                if len(da.x) > 1 and len(da.y) > 1:
                    x_res = float(da.x[1] - da.x[0])
                    y_res = float(da.y[0] - da.y[1])
                    return from_origin(
                        float(da.x[0]) - x_res / 2,
                        float(da.y[0]) + y_res / 2,
                        x_res,
                        y_res
                    )
        except:
            pass
        return None

    @staticmethod
    def _create_simple_tiff(arr2d, nodata):
        """创建简单TIFF"""
        try:
            profile = {
                'driver': 'GTiff',
                'dtype': rasterio.float32,
                'count': 1,
                'height': arr2d.shape[0],
                'width': arr2d.shape[1],
                'transform': from_origin(0, arr2d.shape[0], 1, 1),
                'compress': 'lzw',
                'nodata': nodata
            }

            memfile = MemoryFile()
            with memfile.open(**profile) as dst:
                dst.write(arr2d, 1)

            data = memfile.read()
            memfile.close()
            return data
        except Exception as e:
            logger.error(f"Simple TIFF creation failed: {e}")
            return b''
# reports/generator.py
"""
报告生成模块
"""

import datetime
import numpy as np
import xarray as xr

from utils import logger


class ReportGenerator:
    """分析报告生成器"""

    @staticmethod
    def generate_text_report(analysis_results, data_info, output_path):
        """生成文本格式报告

        Args:
            analysis_results: 分析结果字典
            data_info: 数据信息字典
            output_path: 输出路径

        Returns:
            str: 输出文件路径
        """
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write("=" * 80 + "\n")
                f.write("时序遥感分析报告\n")
                f.write("=" * 80 + "\n\n")

                # 基本信息
                f.write("【基本信息】\n")
                f.write(f"生成时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"数据时间范围: {data_info.get('time_range', 'N/A')}\n")
                f.write(f"时间序列长度: {data_info.get('n_time', 'N/A')} 期\n")
                f.write(f"空间大小: {data_info.get('ny', 'N/A')} × {data_info.get('nx', 'N/A')} 像元\n")
                f.write(f"数据频率: {data_info.get('frequency', 'N/A')}\n\n")

                # 分析结果
                for analysis_name, results in analysis_results.items():
                    f.write("-" * 80 + "\n")
                    f.write(f"【{analysis_name} 分析结果】\n")
                    f.write("-" * 80 + "\n")

                    if isinstance(results, dict):
                        for key, data_array in results.items():
                            stats = ReportGenerator._calculate_statistics(data_array)
                            f.write(f"\n{key}:\n")
                            f.write(f"  最小值: {stats['min']:.6f}\n")
                            f.write(f"  最大值: {stats['max']:.6f}\n")
                            f.write(f"  平均值: {stats['mean']:.6f}\n")
                            f.write(f"  标准差: {stats['std']:.6f}\n")
                            f.write(f"  有效像元数: {stats['valid_count']:,}\n")
                    else:
                        stats = ReportGenerator._calculate_statistics(results)
                        f.write(f"  最小值: {stats['min']:.6f}\n")
                        f.write(f"  最大值: {stats['max']:.6f}\n")
                        f.write(f"  平均值: {stats['mean']:.6f}\n")
                        f.write(f"  标准差: {stats['std']:.6f}\n")
                        f.write(f"  有效像元数: {stats['valid_count']:,}\n")

                    f.write("\n")

                f.write("=" * 80 + "\n")
                f.write("报告结束\n")
                f.write("=" * 80 + "\n")

            logger.info(f"Text report generated: {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"Text report generation failed: {e}")
            raise

    @staticmethod
    def _calculate_statistics(data_array):
        """计算统计信息

        Args:
            data_array: 数据数组

        Returns:
            dict: 统计信息字典
        """
        values = data_array.values if isinstance(data_array, xr.DataArray) else np.array(data_array)

        # 转为2D
        if values.ndim > 2:
            values = np.nanmean(values, axis=0)

        valid = values[~np.isnan(values)]

        return {
            'min': float(np.min(valid)) if len(valid) > 0 else np.nan,
            'max': float(np.max(valid)) if len(valid) > 0 else np.nan,
            'mean': float(np.mean(valid)) if len(valid) > 0 else np.nan,
            'std': float(np.std(valid)) if len(valid) > 0 else np.nan,
            'valid_count': int(len(valid))
        }
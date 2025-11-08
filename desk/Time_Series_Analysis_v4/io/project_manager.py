# io/project_manager.py
"""
项目管理模块
"""

import pickle
import json
import datetime

from config import Config
from utils import logger


class ProjectManager:
    """项目管理器"""

    @staticmethod
    def save_project(data_stack, analysis_results, parameters, file_path):
        """保存完整项目

        Args:
            data_stack: 数据栈
            analysis_results: 分析结果
            parameters: 参数配置
            file_path: 保存路径

        Returns:
            bool: 是否成功
        """
        try:
            project_data = {
                'version': Config.VERSION,
                'timestamp': datetime.datetime.now().isoformat(),
                'data_stack': data_stack,
                'analysis_results': analysis_results,
                'parameters': parameters,
                'metadata': {
                    'n_time': len(data_stack.time) if data_stack is not None else 0,
                    'shape': data_stack.shape if data_stack is not None else None,
                }
            }

            with open(file_path, 'wb') as f:
                pickle.dump(project_data, f, protocol=pickle.HIGHEST_PROTOCOL)

            logger.info(f"Project saved: {file_path}")
            return True

        except Exception as e:
            logger.error(f"Project save failed: {e}")
            raise

    @staticmethod
    def load_project(file_path):
        """加载项目

        Args:
            file_path: 项目文件路径

        Returns:
            dict: 项目数据
        """
        try:
            with open(file_path, 'rb') as f:
                project_data = pickle.load(f)

            logger.info(f"Project loaded: {file_path}")
            return project_data

        except Exception as e:
            logger.error(f"Project load failed: {e}")
            raise

    @staticmethod
    def export_parameters(parameters, file_path):
        """导出参数配置为JSON

        Args:
            parameters: 参数字典
            file_path: 输出路径

        Returns:
            bool: 是否成功
        """
        try:
            # 转换不可序列化的对象
            serializable_params = {}
            for key, value in parameters.items():
                if isinstance(value, (str, int, float, bool, list, dict, type(None))):
                    serializable_params[key] = value
                else:
                    serializable_params[key] = str(value)

            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(serializable_params, f, indent=2, ensure_ascii=False)

            logger.info(f"Parameters exported: {file_path}")
            return True

        except Exception as e:
            logger.error(f"Parameter export failed: {e}")
            raise

    @staticmethod
    def import_parameters(file_path):
        """导入参数配置

        Args:
            file_path: 参数文件路径

        Returns:
            dict: 参数字典
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                parameters = json.load(f)

            logger.info(f"Parameters imported: {file_path}")
            return parameters

        except Exception as e:
            logger.error(f"Parameter import failed: {e}")
            raise
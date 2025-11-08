# utils/progress.py
"""
进度跟踪模块
"""

from utils.logger_config import logger


class ProgressTracker:
    """进度跟踪器"""

    def __init__(self, total_steps=100):
        self.total_steps = total_steps
        self.current_step = 0
        self.callbacks = []
        self.is_cancelled = False

    def update(self, step_name="", progress=None):
        """更新进度

        Args:
            step_name: 步骤名称
            progress: 进度值(0-100)，如果为None则自增
        """
        if progress is not None:
            self.current_step = progress
        else:
            self.current_step += 1

        percentage = min(100, (self.current_step / self.total_steps) * 100)

        for callback in self.callbacks:
            try:
                callback(step_name, percentage)
            except Exception as e:
                logger.error(f"Progress callback error: {e}")

    def add_callback(self, callback):
        """添加回调函数"""
        self.callbacks.append(callback)

    def cancel(self):
        """取消操作"""
        self.is_cancelled = True
        logger.info("Operation cancelled by user")

    def reset(self):
        """重置"""
        self.current_step = 0
        self.is_cancelled = False
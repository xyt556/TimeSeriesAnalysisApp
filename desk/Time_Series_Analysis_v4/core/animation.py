# core/animation.py
"""
时序动画生成模块
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from config import Config
from utils import logger


class AnimationGenerator:
    """动画生成器"""

    @staticmethod
    def create_timeseries_animation(stack, output_path,
                                    fps=None, cmap='viridis', dpi=100,
                                    title_template="时间: {time}",
                                    progress_callback=None):
        """生成时序动画

        Args:
            stack: 时间序列数据
            output_path: 输出路径 (.gif 或 .mp4)
            fps: 帧率
            cmap: 颜色映射
            dpi: 分辨率
            title_template: 标题模板
            progress_callback: 进度回调函数

        Returns:
            输出文件路径
        """
        if fps is None:
            fps = Config.ANIMATION_FPS

        logger.info(f"Creating animation: {output_path}")

        times = stack.time.values
        n_frames = len(times)

        # 计算全局vmin/vmax以保持颜色一致性
        vmin = float(np.nanpercentile(stack.values, 2))
        vmax = float(np.nanpercentile(stack.values, 98))

        fig, ax = plt.subplots(figsize=(10, 8))

        def init():
            ax.clear()
            return []

        def animate(frame):
            ax.clear()
            data = stack.isel(time=frame).values

            im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)

            # 格式化时间标签
            try:
                time_str = pd.to_datetime(str(times[frame])).strftime('%Y-%m-%d')
            except:
                time_str = str(times[frame])

            title = title_template.format(time=time_str, frame=frame + 1, total=n_frames)
            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.axis('off')

            # 添加色标
            if frame == 0:
                cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                cbar.set_label('值', rotation=270, labelpad=15)

            if progress_callback:
                progress_callback("生成动画帧", ((frame + 1) / n_frames) * 100)

            return [im]

        try:
            anim = animation.FuncAnimation(
                fig, animate, init_func=init,
                frames=n_frames, interval=1000 / fps, blit=False
            )

            # 保存动画
            if output_path.endswith('.gif'):
                writer = animation.PillowWriter(fps=fps)
                anim.save(output_path, writer=writer, dpi=dpi)
            elif output_path.endswith('.mp4'):
                writer = animation.FFMpegWriter(fps=fps, bitrate=1800)
                anim.save(output_path, writer=writer, dpi=dpi)
            else:
                raise ValueError("输出格式必须是 .gif 或 .mp4")

            plt.close(fig)
            logger.info(f"Animation saved: {output_path}")
            return output_path

        except Exception as e:
            plt.close(fig)
            logger.error(f"Animation creation failed: {e}")
            raise
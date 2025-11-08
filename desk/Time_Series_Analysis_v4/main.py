# main.py
"""
程序入口文件
"""

import sys
import os
import warnings

# 添加当前目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# 忽略警告
warnings.filterwarnings('ignore')

# 导入主窗口
try:
    from rs_ui import MainWindow
    from utils import logger
    from config import Config
except ImportError as e:
    print(f"导入错误: {e}")
    print(f"当前工作目录: {os.getcwd()}")
    print(f"Python路径: {sys.path}")
    sys.exit(1)


def main():
    """主函数"""
    try:
        logger.info("=" * 60)
        logger.info(f"{Config.APP_NAME} V{Config.VERSION} Starting...")
        logger.info("=" * 60)

        # 创建并运行主窗口
        app = MainWindow()
        app.run()

        logger.info(f"{Config.APP_NAME} V{Config.VERSION} Closed")

    except Exception as e:
        logger.error(f"Application startup failed: {e}", exc_info=True)
        print(f"启动失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
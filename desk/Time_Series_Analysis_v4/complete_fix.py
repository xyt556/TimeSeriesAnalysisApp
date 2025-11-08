# complete_fix.py
"""
完整修复脚本 - 一键解决所有导入问题
"""

import os
import sys

BASE_DIR = r"D:\xyt\soft\TimeSeriesAnalysisApp\desk\Time_Series_Analysis_v4"
os.chdir(BASE_DIR)

print("=" * 80)
print("完整诊断和修复")
print("=" * 80)
print(f"工作目录: {BASE_DIR}\n")

# ========== 第1步：检查文件夹结构 ==========
print("步骤1: 检查文件夹结构")
print("-" * 80)

required_folders = ['config', 'utils', 'core', 'io', 'visualization', 'ui', 'reports']
missing_folders = []

for folder in required_folders:
    folder_path = os.path.join(BASE_DIR, folder)
    exists = os.path.exists(folder_path)
    is_dir = os.path.isdir(folder_path) if exists else False

    if exists and is_dir:
        print(f"✓ {folder}/ 存在")
    else:
        print(f"❌ {folder}/ 不存在或不是文件夹")
        missing_folders.append(folder)
        # 创建文件夹
        os.makedirs(folder_path, exist_ok=True)
        print(f"  → 已创建 {folder}/")

# ========== 第2步：创建所有 __init__.py 文件 ==========
print("\n步骤2: 创建/更新 __init__.py 文件")
print("-" * 80)

init_files = {
    'config/__init__.py': '''# config/__init__.py
"""配置模块"""
from .settings import Config
__all__ = ['Config']
''',

    'utils/__init__.py': '''# utils/__init__.py
"""工具模块"""
from .logger_config import logger, setup_logger
from .progress import ProgressTracker
from .time_utils import TimeExtractor
__all__ = ['logger', 'setup_logger', 'ProgressTracker', 'TimeExtractor']
''',

    'core/__init__.py': '''# core/__init__.py
"""核心分析模块"""
from .analyzers import TrendAnalyzer, BreakpointDetector, FrequencyAnalyzer, STLDecomposer
from .preprocessors import DataPreprocessor
from .clustering import TimeSeriesClusterer
from .animation import AnimationGenerator
__all__ = ['TrendAnalyzer', 'BreakpointDetector', 'FrequencyAnalyzer', 'STLDecomposer', 
           'DataPreprocessor', 'TimeSeriesClusterer', 'AnimationGenerator']
''',

    'io/__init__.py': '''# io/__init__.py
"""输入输出模块"""
from .data_loader import DataLoader
from .exporter import DataExporter
from .project_manager import ProjectManager
__all__ = ['DataLoader', 'DataExporter', 'ProjectManager']
''',

    'visualization/__init__.py': '''# visualization/__init__.py
"""可视化模块"""
from .visualizer import Visualizer
from .interactive import InteractiveTools
__all__ = ['Visualizer', 'InteractiveTools']
''',

    'reports/__init__.py': '''# reports/__init__.py
"""报告生成模块"""
from .generator import ReportGenerator
__all__ = ['ReportGenerator']
''',

    'ui/__init__.py': '''# ui/__init__.py
"""UI模块"""
from .main_window import MainWindow
__all__ = ['MainWindow']
'''
}

for file_path, content in init_files.items():
    full_path = os.path.join(BASE_DIR, file_path)

    # 写入文件
    with open(full_path, 'w', encoding='utf-8') as f:
        f.write(content)

    size = os.path.getsize(full_path)
    print(f"✓ {file_path} ({size} bytes)")

# ========== 第3步：检查关键Python文件 ==========
print("\n步骤3: 检查关键Python文件")
print("-" * 80)

critical_files = {
    'ui/main_window.py': 'MainWindow类定义',
    'config/settings.py': 'Config类定义',
    'utils/logger_config.py': 'logger配置',
}

missing_files = []
for file_path, description in critical_files.items():
    full_path = os.path.join(BASE_DIR, file_path)
    if os.path.exists(full_path):
        size = os.path.getsize(full_path)
        print(f"✓ {file_path} ({size} bytes) - {description}")
    else:
        print(f"❌ {file_path} - 缺失! ({description})")
        missing_files.append(file_path)

# ========== 第4步：测试导入 ==========
print("\n步骤4: 测试模块导入")
print("-" * 80)

# 确保当前目录在Python路径中
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

test_imports = [
    ('config', 'Config', '配置'),
    ('utils', 'logger', '日志'),
    ('utils', 'ProgressTracker', '进度跟踪'),
    ('ui', 'MainWindow', '主窗口'),
]

import_errors = []
for module_name, class_name, description in test_imports:
    try:
        module = __import__(module_name, fromlist=[class_name])
        obj = getattr(module, class_name)
        print(f"✓ from {module_name} import {class_name} - {description}")
    except ImportError as e:
        print(f"❌ from {module_name} import {class_name} - 导入失败: {e}")
        import_errors.append((module_name, class_name, str(e)))
    except AttributeError as e:
        print(f"❌ from {module_name} import {class_name} - 属性错误: {e}")
        import_errors.append((module_name, class_name, str(e)))
    except Exception as e:
        print(f"⚠️  from {module_name} import {class_name} - 其他错误: {e}")
        import_errors.append((module_name, class_name, str(e)))

# ========== 第5步：生成详细报告 ==========
print("\n" + "=" * 80)
print("诊断报告")
print("=" * 80)

if missing_folders:
    print(f"\n⚠️  缺少的文件夹 ({len(missing_folders)}):")
    for f in missing_folders:
        print(f"  - {f}")
    print("  → 已自动创建")

if missing_files:
    print(f"\n❌ 缺少的关键文件 ({len(missing_files)}):")
    for f in missing_files:
        print(f"  - {f}")
    print("\n  请确保这些文件已从提供的代码中创建！")

if import_errors:
    print(f"\n❌ 导入错误 ({len(import_errors)}):")
    for module, cls, error in import_errors:
        print(f"  - {module}.{cls}: {error}")

# ========== 第6步：生成解决方案 ==========
print("\n" + "=" * 80)

if not missing_files and not import_errors:
    print("✅ 所有检查通过！")
    print("\n现在可以运行:")
    print("  python main.py")
else:
    print("❌ 仍有问题需要解决\n")

    if missing_files:
        print("解决方案 1: 缺失的Python文件")
        print("-" * 60)
        for file_path in missing_files:
            print(f"\n需要创建: {file_path}")
            if file_path == 'ui/main_window.py':
                print("这是最关键的文件！")
                print("建议操作:")
                print("1. 使用我提供的 app_single_file.py 测试环境")
                print("2. 或者重新复制 ui/main_window.py 的代码")

    if import_errors:
        print("\n解决方案 2: 导入错误")
        print("-" * 60)
        print("可能的原因:")
        print("1. Python文件中有语法错误")
        print("2. 缺少依赖的类或函数定义")
        print("3. 文件编码问题")
        print("\n建议:")
        print("1. 先运行单文件版本测试: python app_single_file.py")
        print("2. 检查每个.py文件的语法")

print("=" * 80)

# ========== 第7步：创建验证脚本 ==========
print("\n生成验证脚本...")

verify_script = '''# verify_structure.py
"""验证项目结构"""
import os

base = r"D:\\xyt\\soft\\TimeSeriesAnalysisApp\\desk\\Time_Series_Analysis_v4"

structure = {
    'ui': ['__init__.py', 'main_window.py', 'components.py', 'dialogs.py'],
    'config': ['__init__.py', 'settings.py'],
    'utils': ['__init__.py', 'logger_config.py', 'progress.py', 'time_utils.py'],
    'core': ['__init__.py', 'analyzers.py', 'preprocessors.py', 'clustering.py', 'animation.py'],
    'io': ['__init__.py', 'data_loader.py', 'exporter.py', 'project_manager.py'],
    'visualization': ['__init__.py', 'visualizer.py', 'interactive.py'],
    'reports': ['__init__.py', 'generator.py'],
}

print("项目结构验证:")
print("=" * 60)

total = 0
existing = 0

for folder, files in structure.items():
    print(f"\\n{folder}/")
    for file in files:
        path = os.path.join(base, folder, file)
        total += 1
        if os.path.exists(path):
            size = os.path.getsize(path)
            print(f"  ✓ {file} ({size} bytes)")
            existing += 1
        else:
            print(f"  ❌ {file}")

print(f"\\n完成度: {existing}/{total} ({existing/total*100:.1f}%)")

if existing == total:
    print("\\n✅ 所有文件完整！可以运行 python main.py")
else:
    print(f"\\n❌ 缺少 {total-existing} 个文件")
'''

with open(os.path.join(BASE_DIR, 'verify_structure.py'), 'w', encoding='utf-8') as f:
    f.write(verify_script)

print("✓ 已生成 verify_structure.py")
print("\n运行 python verify_structure.py 可以验证文件完整性")
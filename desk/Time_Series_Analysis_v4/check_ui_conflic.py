# check_ui_conflict.py
"""
检查是否存在ui包冲突
"""

import sys
import importlib.util

print("=" * 80)
print("检查 ui 模块冲突")
print("=" * 80)

# 方法1：尝试导入ui
print("\n1. 尝试导入 ui 模块...")
try:
    import ui

    print(f"✓ 找到 ui 模块")
    print(f"  位置: {ui.__file__ if hasattr(ui, '__file__') else 'unknown'}")
    print(f"  路径: {ui.__path__ if hasattr(ui, '__path__') else 'N/A'}")
    print(f"  包: {ui.__package__ if hasattr(ui, '__package__') else 'N/A'}")

    # 检查是否有MainWindow
    if hasattr(ui, 'MainWindow'):
        print(f"  ✓ 包含 MainWindow")
    else:
        print(f"  ❌ 不包含 MainWindow")
        print(f"  可用属性: {dir(ui)}")

except ImportError as e:
    print(f"❌ 无法导入 ui: {e}")

# 方法2：查找所有ui模块
print("\n2. 搜索所有 ui 相关模块...")
for path in sys.path:
    print(f"\n检查路径: {path}")
    try:
        import os

        if os.path.isdir(path):
            for item in os.listdir(path):
                if item.startswith('ui') and (item.endswith('.py') or os.path.isdir(os.path.join(path, item))):
                    print(f"  找到: {item}")
    except:
        pass

# 方法3：检查已安装的包
print("\n3. 检查已安装的 ui 相关包...")
try:
    import pkg_resources

    for dist in pkg_resources.working_set:
        if 'ui' in dist.project_name.lower():
            print(f"  {dist.project_name} {dist.version}")
except:
    print("  无法检查（pkg_resources不可用）")

print("\n" + "=" * 80)
print("建议:")
print("1. 如果找到了其他的ui模块，建议重命名项目的ui文件夹为rs_ui")
print("2. 运行: python rename_ui_module.py")
print("=" * 80)
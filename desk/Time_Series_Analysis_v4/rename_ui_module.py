# rename_ui_module.py
"""
将ui模块重命名为rs_ui以避免命名冲突
"""

import os
import shutil

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

print("=" * 80)
print("重命名 ui 模块为 rs_ui")
print("=" * 80)

# 步骤1：重命名文件夹
old_ui_path = os.path.join(BASE_DIR, 'ui')
new_ui_path = os.path.join(BASE_DIR, 'rs_ui')

if os.path.exists(old_ui_path):
    if os.path.exists(new_ui_path):
        print(f"删除旧的 rs_ui 文件夹...")
        shutil.rmtree(new_ui_path)

    print(f"重命名: ui -> rs_ui")
    shutil.move(old_ui_path, new_ui_path)
    print("✓ 文件夹重命名成功")
else:
    print("⚠️  ui 文件夹不存在")
    if os.path.exists(new_ui_path):
        print("✓ rs_ui 文件夹已存在")
    else:
        print("❌ 两个文件夹都不存在！")
        exit(1)

# 步骤2：更新 main.py
main_file = os.path.join(BASE_DIR, 'main.py')
print(f"\n更新 {main_file}...")

with open(main_file, 'r', encoding='utf-8') as f:
    content = f.read()

# 替换导入语句
content = content.replace('from ui import MainWindow', 'from rs_ui import MainWindow')

with open(main_file, 'w', encoding='utf-8') as f:
    f.write(content)

print("✓ main.py 已更新")

# 步骤3：检查其他文件中的ui导入
print("\n检查其他文件...")
files_to_check = [
    'rs_ui/dialogs.py',
    'rs_ui/main_window.py',
]

for file_path in files_to_check:
    full_path = os.path.join(BASE_DIR, file_path)
    if os.path.exists(full_path):
        with open(full_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # 检查是否需要更新
        if 'from ui.' in content or 'from .ui' in content:
            print(f"  需要更新: {file_path}")
            content = content.replace('from ui.', 'from rs_ui.')
            content = content.replace('from .ui', 'from .rs_ui')

            with open(full_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"  ✓ 已更新")
        else:
            print(f"  - 无需更新: {file_path}")

print("\n" + "=" * 80)
print("✅ 重命名完成！")
print("\n现在运行:")
print("  python main.py")
print("=" * 80)
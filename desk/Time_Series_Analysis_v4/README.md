# 时序遥感分析系统 V4.0

一个功能全面的时序遥感数据分析工具，支持趋势分析、突变检测、周期分析、数据预处理、聚类分析等功能。

## 功能特性

### 核心分析
- **Theil-Sen趋势分析**: 稳健的长期趋势估计
- **Mann-Kendall检验**: 趋势显著性检验
- **BFAST突变检测**: 时间序列结构突变识别
- **FFT周期分析**: 周期性特征提取
- **STL分解**: 趋势-季节-残差分解

### 高级功能
- **数据预处理**: Savitzky-Golay平滑、异常值检测、数据插值
- **时间序列聚类**: K-means、层次聚类分析
- **动画生成**: GIF/MP4时序动画导出
- **交互式地图**: 像元点击查询
- **项目管理**: 保存/加载完整分析项目
- **报告生成**: 自动生成分析报告



## 安装说明

### 环境要求
- Python 3.8+
- 建议使用 Anaconda/Miniconda

### 依赖安装

```bash
# 创建虚拟环境（推荐）
conda create -n rs_analysis python=3.9
conda activate rs_analysis

# 安装依赖
pip install -r requirements.txt

使用指南
1. 数据准备
格式: GeoTIFF (.tif, .tiff)
时间信息: 文件名必须包含可解析的时间信息
年度数据: NDVI_2000.tif, NDVI_2001.tif
月度数据: NDVI_200001.tif, NDVI_2000_01.tif
日期数据: NDVI_2000_001.tif (年_儒略日)
所有文件必须具有相同的空间范围和分辨率
2. 快速开始
启动程序
菜单 → 文件 → 打开数据文件
选择时序GeoTIFF文件
点击"加载数据"
选择分析方法并设置参数
点击"执行分析"
查看结果并导出
3. 数据预处理
菜单 → 数据处理 → 数据平滑
选择Savitzky-Golay或移动平均方法
设置参数并执行
4. 高级分析
菜单 → 高级分析 → 时间序列聚类
菜单 → 高级分析 → 生成时序动画
5. 项目管理
菜单 → 文件 → 保存项目
菜单 → 文件 → 加载项目
技术支持
版本: 3.0
开发: @3S&ML
日志文件: rs_analysis.log
许可证
MIT License

更新日志
V3.0 (2024)
全面重构为模块化架构
新增聚类分析功能
新增动画生成功能
改进数据预处理
优化性能和用户体验


---

## 🎉 完整模块化代码已全部提供！

### 📦 文件清单（共27个文件）

**配置模块 (2个文件)**
1. `config/settings.py`
2. `config/__init__.py`

**工具模块 (4个文件)**
3. `utils/logger_config.py`
4. `utils/progress.py`
5. `utils/time_utils.py`
6. `utils/__init__.py`

**核心分析模块 (5个文件)**
7. `core/analyzers.py`
8. `core/preprocessors.py`
9. `core/clustering.py`
10. `core/animation.py`
11. `core/__init__.py`

**输入输出模块 (4个文件)**
12. `io/data_loader.py`
13. `io/exporter.py`
14. `io/project_manager.py`
15. `io/__init__.py`

**可视化模块 (3个文件)**
16. `visualization/visualizer.py`
17. `visualization/interactive.py`
18. `visualization/__init__.py`

**报告模块 (2个文件)**
19. `reports/generator.py`
20. `reports/__init__.py`

**UI模块 (4个文件)**
21. `ui/components.py`
22. `ui/dialogs.py`
23. `ui/main_window.py`
24. `ui/__init__.py`

**主程序 (3个文件)**
25. `main.py`
26. `requirements.txt`
27. `README.md`

### 🚀 运行步骤

1. **创建项目目录结构**
```bash
mkdir -p remote_sensing_analysis/{config,core,utils,io,visualization,ui,reports}

复制所有代码文件到对应目录

安装依赖

Bash

pip install -r requirements.txt
运行程序
Bash

python main.py
✨ 模块化优势
清晰的代码组织: 每个模块职责单一
易于维护: 修改某个功能只需修改对应模块
便于扩展: 新增功能只需添加新模块
代码复用: 各模块可独立导入使用
团队协作: 不同人员可负责不同模块
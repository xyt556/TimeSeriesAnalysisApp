# app.py
import streamlit as st
import tempfile
from pathlib import Path
import re
import xarray as xr
import rioxarray as rxr
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import warnings
import datetime

warnings.filterwarnings('ignore')

# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'STHeiti']
matplotlib.rcParams['axes.unicode_minus'] = False

from utils.analysis_tools import (
    theil_sen_trend,
    mann_kendall_test,
    bfast_detection,
    fft_analysis,
    stl_decompose_pixelwise
)
from utils.visualization import (
    plot_map,
    plot_pixel_timeseries,
    dataarray_to_bytes_tif,
    fig_to_bytes_png
)

# ==================== 中文字体配置 ====================

# ==================== 中文字体配置 (优化版，适用于云端部署) ====================

def configure_chinese_fonts():
    """
    配置matplotlib中文字体，优先使用项目内置字体文件，确保云端部署时中文正常显示
    """
    import platform
    from matplotlib.font_manager import fontManager, FontProperties
    import os

    # ===== 策略1：优先加载项目内置字体（适用于部署环境） =====
    font_filename = 'SIMSUN.TTC'
    font_path = os.path.join('fonts', font_filename)

    if os.path.exists(font_path):
        try:
            # 动态注册字体到matplotlib
            fontManager.addfont(font_path)

            # 获取字体的实际名称
            prop = FontProperties(fname=font_path)
            font_name = prop.get_name()  # 通常是 'Source Han Sans SC'

            # 设置为matplotlib的默认字体
            plt.rcParams['font.sans-serif'] = [font_name]
            plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

            return True, font_name
        except Exception as e:
            print(f"⚠️ 加载内置字体失败: {e}")
    else:
        print(f"⚠️ 未找到字体文件: {font_path}")

    # ===== 策略2：回退到系统字体（适用于本地开发） =====
    system = platform.system()
    chinese_fonts = []

    if system == 'Windows':
        chinese_fonts = ['SimHei', 'Microsoft YaHei', 'SimSun']
    elif system == 'Darwin':  # macOS
        chinese_fonts = ['PingFang SC', 'Heiti SC', 'STHeiti']
    else:  # Linux
        chinese_fonts = ['WenQuanYi Micro Hei', 'WenQuanYi Zen Hei', 'Droid Sans Fallback']

    from matplotlib.font_manager import FontManager
    fm = FontManager()
    available_fonts = {f.name for f in fm.ttflist}

    selected_font = None
    for font in chinese_fonts:
        if font in available_fonts:
            selected_font = font
            break

    # 如果仍然找不到，尝试搜索包含中文关键词的字体
    if selected_font is None:
        for font in available_fonts:
            if any(keyword in font.lower() for keyword in ['chinese', 'cjk', 'han', 'hei', 'song']):
                selected_font = font
                break

    if selected_font:
        plt.rcParams['font.sans-serif'] = [selected_font]
        plt.rcParams['axes.unicode_minus'] = False
        return True, selected_font
    else:
        print("❌ 未找到任何可用的中文字体")
        return False, None


# 执行字体配置
CHINESE_SUPPORT, SELECTED_FONT = configure_chinese_fonts()

# 页面配置
st.set_page_config(
    page_title="时序遥感分析系统",
    layout="wide",
    page_icon="🛰️"
)

st.title("🛰️ 时序遥感分析系统")
st.markdown("""
**功能模块：** Theil–Sen趋势分析 | Mann–Kendall检验 | BFAST突变检测 | FFT周期分析 | STL分解
""")


# ---------------------------
# 文件上传与预处理
# ---------------------------
@st.cache_data(show_spinner=False)
def extract_time(filename):
    """
    从文件名中提取时间信息，支持多种格式
    返回 datetime 对象
    """
    # 尝试匹配 YYYY_DDD 格式 (2020_001, 2020_365)
    m = re.search(r'(19\d{2}|20\d{2})_(\d{3})', filename)
    if m:
        year = int(m.group(1))
        day_of_year = int(m.group(2))
        # 将一年中的第几天转换为月份和日期
        try:
            date = datetime.datetime(year, 1, 1) + datetime.timedelta(days=day_of_year - 1)
            return date
        except:
            return datetime.datetime(year, 1, 1)

    # 尝试匹配 YYYY_MM 格式 (2020_01, 2020_12)
    m = re.search(r'(19\d{2}|20\d{2})_(\d{1,2})', filename)
    if m:
        year = int(m.group(1))
        month = int(m.group(2))
        return datetime.datetime(year, month, 1)

    # 尝试匹配 YYYYMM 格式 (202001, 202012)
    m = re.search(r'(19\d{2}|20\d{2})(\d{2})', filename)
    if m:
        year = int(m.group(1))
        month = int(m.group(2))
        return datetime.datetime(year, month, 1)

    # 尝试匹配 YYYY 格式 (2000, 2001)
    m = re.search(r'(19\d{2}|20\d{2})', filename)
    if m:
        year = int(m.group(0))
        return datetime.datetime(year, 1, 1)  # 年度数据默认设为1月1日

    # 尝试匹配英文月份
    month_map = {
        'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
        'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
    }

    for month_name, month_num in month_map.items():
        if month_name in filename:
            m = re.search(r'(19\d{2}|20\d{2})', filename)
            if m:
                year = int(m.group(0))
                return datetime.datetime(year, month_num, 1)

    return None


@st.cache_data(show_spinner=False)
def load_and_stack_files(_uploaded_files):
    """加载并堆叠文件 - 修复时间坐标问题"""
    tmpdir = Path(tempfile.mkdtemp())
    paths = []

    for f in _uploaded_files:
        p = tmpdir / f.name
        p.write_bytes(f.getbuffer())
        paths.append(p)

    # 提取时间信息
    times = [extract_time(f.name) for f in _uploaded_files]

    # 显示文件名和时间提取结果用于调试
    st.info("文件名和时间提取结果:")
    for f, t in zip(_uploaded_files, times):
        if t:
            st.write(f"- {f.name} -> {t}")
        else:
            st.write(f"- {f.name} -> 无法提取时间")

    # 检查时间提取
    invalid_files = [(f.name, t) for f, t in zip(_uploaded_files, times) if t is None]
    if invalid_files:
        st.error("以下文件中未检测到有效时间信息:")
        for fname, time_val in invalid_files:
            st.error(f"  - {fname}")
        st.info("💡 支持的文件名格式:")
        st.info("   - 年度数据: NDVI_2000.tif, NDVI_2001_徐州.tif")
        st.info("   - 月度数据: NDVI_200001.tif, NDVI_2000_01.tif, NDVI_2000_01_徐州.tif")
        st.info("   - 日度数据: NDVI_2000_001.tif, NDVI_2000_365_徐州.tif")
        return None

    # 按时间排序并检查重复
    sorted_indices = sorted(range(len(times)), key=lambda i: times[i])
    paths = [paths[i] for i in sorted_indices]
    times = [times[i] for i in sorted_indices]
    _uploaded_files = [_uploaded_files[i] for i in sorted_indices]

    # 检查时间重复并显示详细信息
    time_count = {}
    duplicate_files = {}
    for f, t in zip(_uploaded_files, times):
        if t not in time_count:
            time_count[t] = []
        time_count[t].append(f.name)

    duplicate_times = [t for t, files in time_count.items() if len(files) > 1]

    if duplicate_times:
        st.warning("⚠️ 检测到重复的时间点:")
        for t in duplicate_times:
            st.error(f"时间 {t.strftime('%Y-%m-%d')} 对应的文件:")
            for fname in time_count[t]:
                st.error(f"  - {fname}")
        st.info("时序分析要求每个时间点只有一个观测值，请检查文件命名。")

        # 询问用户是否继续
        continue_anyway = st.checkbox("忽略重复时间点，继续分析", value=False)
        if not continue_anyway:
            return None

    # 读取数据
    data_list = []
    time_coords = []
    for p, t in zip(paths, times):
        try:
            da = rxr.open_rasterio(str(p), chunks={'x': 512, 'y': 512}).squeeze()
            if "band" in da.dims:
                da = da.isel(band=0).drop_vars('band')

            # 确保数据是2D的 (y, x)
            if da.ndim != 2:
                st.error(f"文件 {p.name} 的维度不正确，期望2D数据 (y, x)，实际维度: {da.dims}")
                continue

            data_list.append(da)
            time_coords.append(t)

        except Exception as e:
            st.error(f"读取文件 {p.name} 时出错: {e}")
            continue

    if not data_list:
        st.error("没有成功读取任何文件")
        return None

    # 堆叠数据
    try:
        # 使用concat而不是expand_dims，确保时间维度正确
        stack = xr.concat(data_list, dim="time")
        stack = stack.assign_coords(time=time_coords)
        stack = stack.transpose('time', 'y', 'x')

        # 验证数据形状
        st.info(f"数据栈形状: {stack.shape} (时间, Y, X)")

        # 显示时间坐标信息
        time_info = []
        for t in stack.time.values:
            if isinstance(t, np.datetime64):
                time_info.append(np.datetime_as_string(t, unit='D'))
            else:
                time_info.append(str(t))
        st.info(f"时间坐标: {time_info}")

        return stack

    except Exception as e:
        st.error(f"数据堆叠失败: {e}")
        return None

@st.cache_data(show_spinner=False)
def load_and_stack_files(_uploaded_files):
    """加载并堆叠文件 - 修复时间坐标问题"""
    tmpdir = Path(tempfile.mkdtemp())
    paths = []

    for f in _uploaded_files:
        p = tmpdir / f.name
        p.write_bytes(f.getbuffer())
        paths.append(p)

    # 提取时间信息
    times = [extract_time(f.name) for f in _uploaded_files]

    # 检查时间提取
    invalid_files = [(f.name, t) for f, t in zip(_uploaded_files, times) if t is None]
    if invalid_files:
        st.error("以下文件中未检测到有效时间信息:")
        for fname, time_val in invalid_files:
            st.error(f"  - {fname}")
        st.info("💡 支持的文件名格式:")
        st.info("   - 年度数据: NDVI_2000.tif, NDVI_2001_徐州.tif")
        st.info("   - 月度数据: NDVI_200001.tif, NDVI_2000_01.tif, NDVI_2000_01_徐州.tif")
        st.info("   - 日度数据: NDVI_2000_001.tif, NDVI_2000_365_徐州.tif")
        return None

    # 按时间排序并检查重复
    sorted_indices = sorted(range(len(times)), key=lambda i: times[i])
    paths = [paths[i] for i in sorted_indices]
    times = [times[i] for i in sorted_indices]
    _uploaded_files = [_uploaded_files[i] for i in sorted_indices]

    # 检查时间重复并显示详细信息
    time_count = {}
    duplicate_files = {}
    for f, t in zip(_uploaded_files, times):
        if t not in time_count:
            time_count[t] = []
        time_count[t].append(f.name)

    duplicate_times = [t for t, files in time_count.items() if len(files) > 1]

    if duplicate_times:
        st.warning("⚠️ 检测到重复的时间点:")
        for t in duplicate_times:
            st.error(f"时间 {t.strftime('%Y-%m-%d')} 对应的文件:")
            for fname in time_count[t]:
                st.error(f"  - {fname}")
        st.info("时序分析要求每个时间点只有一个观测值，请检查文件命名。")

        # 询问用户是否继续
        continue_anyway = st.checkbox("忽略重复时间点，继续分析", value=False)
        if not continue_anyway:
            return None

    # 读取数据
    data_list = []
    time_coords = []
    for p, t in zip(paths, times):
        try:
            da = rxr.open_rasterio(str(p), chunks={'x': 512, 'y': 512}).squeeze()
            if "band" in da.dims:
                da = da.isel(band=0).drop_vars('band')

            # 确保数据是2D的 (y, x)
            if da.ndim != 2:
                st.error(f"文件 {p.name} 的维度不正确，期望2D数据 (y, x)，实际维度: {da.dims}")
                continue

            data_list.append(da)
            time_coords.append(t)

        except Exception as e:
            st.error(f"读取文件 {p.name} 时出错: {e}")
            continue

    if not data_list:
        st.error("没有成功读取任何文件")
        return None

    # 堆叠数据
    try:
        # 使用concat而不是expand_dims，确保时间维度正确
        stack = xr.concat(data_list, dim="time")
        stack = stack.assign_coords(time=time_coords)
        stack = stack.transpose('time', 'y', 'x')

        # 验证数据形状
        st.info(f"数据栈形状: {stack.shape} (时间, Y, X)")

        # 显示时间坐标信息
        time_info = []
        for t in stack.time.values:
            if isinstance(t, np.datetime64):
                time_info.append(np.datetime_as_string(t, unit='D'))
            else:
                time_info.append(str(t))
        st.info(f"时间坐标: {time_info}")

        return stack

    except Exception as e:
        st.error(f"数据堆叠失败: {e}")
        return None

@st.cache_data(show_spinner=False)
def load_and_stack_files(_uploaded_files):
    """加载并堆叠文件 - 修复时间坐标问题"""
    tmpdir = Path(tempfile.mkdtemp())
    paths = []

    for f in _uploaded_files:
        p = tmpdir / f.name
        p.write_bytes(f.getbuffer())
        paths.append(p)

    # 提取时间信息
    times = [extract_time(f.name) for f in _uploaded_files]

    # 检查时间提取
    invalid_files = [(f.name, t) for f, t in zip(_uploaded_files, times) if t is None]
    if invalid_files:
        st.error("以下文件中未检测到有效时间信息:")
        for fname, time_val in invalid_files:
            st.error(f"  - {fname}")
        st.info("💡 请确保文件名包含4位年份，如: 2000, 1999 或 200001, 200012")
        return None

    # 按时间排序并检查重复
    sorted_indices = sorted(range(len(times)), key=lambda i: times[i])
    paths = [paths[i] for i in sorted_indices]
    times = [times[i] for i in sorted_indices]
    _uploaded_files = [_uploaded_files[i] for i in sorted_indices]

    # 检查时间重复
    unique_times = set()
    duplicate_times = []
    for t in times:
        if t in unique_times:
            duplicate_times.append(t)
        else:
            unique_times.add(t)

    if duplicate_times:
        st.warning(f"⚠️ 检测到重复的时间点: {duplicate_times}")
        st.info("时序分析要求每个时间点只有一个观测值，请检查文件命名。")

    # 读取数据
    data_list = []
    time_coords = []
    for p, t in zip(paths, times):
        try:
            da = rxr.open_rasterio(str(p), chunks={'x': 512, 'y': 512}).squeeze()
            if "band" in da.dims:
                da = da.isel(band=0).drop_vars('band')

            # 确保数据是2D的 (y, x)
            if da.ndim != 2:
                st.error(f"文件 {p.name} 的维度不正确，期望2D数据 (y, x)，实际维度: {da.dims}")
                continue

            data_list.append(da)
            time_coords.append(t)

        except Exception as e:
            st.error(f"读取文件 {p.name} 时出错: {e}")
            continue

    if not data_list:
        st.error("没有成功读取任何文件")
        return None

    # 堆叠数据
    try:
        # 使用concat而不是expand_dims，确保时间维度正确
        stack = xr.concat(data_list, dim="time")
        stack = stack.assign_coords(time=time_coords)
        stack = stack.transpose('time', 'y', 'x')

        # 验证数据形状
        st.info(f"数据栈形状: {stack.shape} (时间, Y, X)")
        st.info(f"时间坐标: {stack.time.values}")

        return stack

    except Exception as e:
        st.error(f"数据堆叠失败: {e}")
        return None


# 初始化session state
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = {}
if 'data_stack' not in st.session_state:
    st.session_state.data_stack = None

# 文件上传区域
st.sidebar.header("📁 数据上传")
uploaded_files = st.sidebar.file_uploader(
    "上传多期 GeoTIFF 文件",
    type=["tif", "tiff"],
    accept_multiple_files=True,
    help="文件名应包含年份，如: NDVI_2000.tif (年度) 或 NDVI_200001.tif (月度)"
)

if uploaded_files:
    # 按文件名排序以确保一致性
    uploaded_files = sorted(uploaded_files, key=lambda f: f.name)

    # 加载数据
    with st.spinner("🔄 正在加载和处理栅格数据..."):
        data_stack = load_and_stack_files(uploaded_files)

    if data_stack is not None:
        st.session_state.data_stack = data_stack

        # 检测数据类型并显示信息
        times = data_stack.time.values
        time_labels = []
        for t in times:
            if isinstance(t, np.datetime64):
                time_labels.append(np.datetime_as_string(t, unit='D'))
            else:
                time_labels.append(str(t))

        # 判断数据频率
        data_frequency = "年度数据"
        if len(times) > 1:
            if isinstance(times[0], np.datetime64):
                time_diff = times[1] - times[0]
                days_diff = time_diff / np.timedelta64(1, 'D')
                if 28 <= days_diff <= 31:
                    data_frequency = "月度数据"
                elif 360 <= days_diff <= 370:
                    data_frequency = "年度数据"

        # 显示数据信息
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("数据频率", data_frequency)
        with col2:
            st.metric("时间序列长度", f"{data_stack.sizes['time']} 期")
        with col3:
            st.metric("空间分辨率", f"{data_stack.sizes['y']} × {data_stack.sizes['x']}")
        with col4:
            if len(time_labels) > 0:
                st.metric("时间范围", f"{time_labels[0]} 至 {time_labels[-1]}")

        # 数据预览
        with st.expander("📊 数据预览", expanded=True):
            tab1, tab2, tab3 = st.tabs(["空间分布", "时间序列抽样", "统计信息"])

            with tab1:
                st.subheader("第一期栅格预览")
                plot_map(data_stack.isel(time=0),
                         title=f"时间: {time_labels[0]}")

            with tab2:
                st.subheader("随机像元时间序列抽样")
                # 随机选择几个像元显示时间序列
                ny, nx = data_stack.sizes['y'], data_stack.sizes['x']
                if ny > 0 and nx > 0:
                    # 随机选择几个位置
                    import random

                    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
                    axes = axes.flatten()

                    for i, ax in enumerate(axes):
                        row = random.randint(0, ny - 1)
                        col = random.randint(0, nx - 1)
                        ts = data_stack[:, row, col].values

                        ax.plot(time_labels, ts, 'o-', markersize=3)
                        ax.set_title(f'像元 ({row}, {col})')
                        ax.set_ylabel('值')
                        ax.tick_params(axis='x', rotation=45)
                        ax.grid(True, alpha=0.3)

                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)

            with tab3:
                st.write("数据统计:")
                stats_data = {
                    '最小值': float(data_stack.min()),
                    '最大值': float(data_stack.max()),
                    '平均值': float(data_stack.mean()),
                    '标准差': float(data_stack.std())
                }
                for key, value in stats_data.items():
                    st.write(f"- {key}: {value:.4f}")

        # ---------------------------
        # 分析控制
        # ---------------------------
        st.sidebar.header("🔧 分析控制")

        # 分析选择
        st.sidebar.subheader("选择分析方法")
        analysis_options = {
            "Theil–Sen 趋势分析": "theilsen",
            "Mann–Kendall 检验": "mk",
            "BFAST 突变检测": "bfast",
            "FFT 周期分析": "fft",
            "STL 分解": "stl"
        }

        selected_analyses = []
        for name, key in analysis_options.items():
            if st.sidebar.checkbox(name, value=True, key=f"checkbox_{key}"):
                selected_analyses.append(key)

        # 根据数据频率设置默认STL周期
        default_stl_period = 12  # 默认月度数据周期
        if data_frequency == "年度数据":
            default_stl_period = 1
            st.sidebar.info("📅 年度数据检测：STL分解可能不适用")

        if 'stl' in selected_analyses:
            stl_period = st.sidebar.number_input(
                "STL 周期参数",
                value=default_stl_period,
                min_value=1,
                max_value=min(24, len(times) // 2),
                help="季节周期长度，月度数据通常为12，年度数据通常为1"
            )

        # 执行按钮
        run_analysis = st.sidebar.button(
            "🚀 执行选中分析",
            type="primary",
            use_container_width=True
        )

        if run_analysis and selected_analyses:
            # 创建进度显示区域
            progress_container = st.container()
            with progress_container:
                st.subheader("分析进度")
                progress_bar = st.progress(0)
                status_text = st.empty()
                percent_text = st.empty()
                time_elapsed_text = st.empty()

            start_time = datetime.datetime.now()

            try:
                total_analyses = len(selected_analyses)
                current_progress = 0


                # 更新进度显示函数
                def update_progress(step_name, progress):
                    progress_bar.progress(progress)
                    percent_text.text(f"进度: {progress * 100:.1f}%")
                    status_text.text(f"🔄 {step_name}")
                    elapsed = datetime.datetime.now() - start_time
                    time_elapsed_text.text(f"已用时: {elapsed.seconds // 60}分{elapsed.seconds % 60}秒")


                # Theil–Sen 分析
                if 'theilsen' in selected_analyses:
                    update_progress("正在计算 Theil–Sen 趋势...", current_progress / total_analyses)
                    slope_da, intercept_da = theil_sen_trend(data_stack)
                    st.session_state.analysis_results['theilsen'] = {
                        'slope': slope_da,
                        'intercept': intercept_da
                    }
                    current_progress += 1
                    update_progress("Theil–Sen 趋势分析完成", current_progress / total_analyses)

                # Mann–Kendall 分析
                if 'mk' in selected_analyses:
                    update_progress("正在计算 Mann–Kendall 检验...", current_progress / total_analyses)
                    mk_da = mann_kendall_test(data_stack)
                    st.session_state.analysis_results['mk'] = mk_da
                    current_progress += 1
                    update_progress("Mann–Kendall 检验完成", current_progress / total_analyses)

                # BFAST 分析
                if 'bfast' in selected_analyses:
                    update_progress("正在检测突变点...", current_progress / total_analyses)
                    break_da = bfast_detection(data_stack)
                    st.session_state.analysis_results['bfast'] = break_da
                    current_progress += 1
                    update_progress("BFAST 突变检测完成", current_progress / total_analyses)

                # FFT 分析
                if 'fft' in selected_analyses:
                    update_progress("正在进行 FFT 周期分析...", current_progress / total_analyses)
                    amp_da, period_da = fft_analysis(data_stack)
                    st.session_state.analysis_results['fft'] = {
                        'amplitude': amp_da,
                        'period': period_da
                    }
                    current_progress += 1
                    update_progress("FFT 周期分析完成", current_progress / total_analyses)

                # STL 分解
                if 'stl' in selected_analyses:
                    if data_frequency == "年度数据":
                        st.warning("⚠️ 年度数据不适合STL分解，季节周期可能无意义")

                    update_progress("正在执行 STL 分解...", current_progress / total_analyses)
                    trend_da, seasonal_da, resid_da = stl_decompose_pixelwise(
                        data_stack,
                        period=stl_period
                    )
                    st.session_state.analysis_results['stl'] = {
                        'trend': trend_da,
                        'seasonal': seasonal_da,
                        'resid': resid_da
                    }
                    current_progress += 1
                    update_progress("STL 分解完成", current_progress / total_analyses)

                # 完成所有分析
                progress_bar.progress(1.0)
                percent_text.text("进度: 100%")
                status_text.text("✅ 所有分析完成!")
                total_time = datetime.datetime.now() - start_time
                time_elapsed_text.text(f"总用时: {total_time.seconds // 60}分{total_time.seconds % 60}秒")
                st.balloons()

                # 短暂延迟后清除进度显示
                import time

                time.sleep(2)
                progress_container.empty()

            except Exception as e:
                progress_bar.progress(1.0)
                percent_text.text("进度: 100%")
                status_text.text("❌ 分析过程中出错")
                st.error(f"错误详情: {e}")

        # ---------------------------
        # 结果显示和下载
        # ---------------------------
        if st.session_state.analysis_results:
            st.header("📋 分析结果")

            # Theil–Sen 结果
            if 'theilsen' in st.session_state.analysis_results:
                with st.expander("📈 Theil–Sen 趋势分析结果", expanded=True):
                    slope_da = st.session_state.analysis_results['theilsen']['slope']
                    col1, col2 = st.columns(2)
                    with col1:
                        plot_map(slope_da, title="Theil–Sen 斜率")
                    with col2:
                        st.download_button(
                            "⬇️ 下载斜率结果 (GeoTIFF)",
                            data=dataarray_to_bytes_tif(slope_da),
                            file_name="theil_sen_slope.tif",
                            mime="image/tiff",
                            use_container_width=True
                        )

            # Mann–Kendall 结果
            if 'mk' in st.session_state.analysis_results:
                with st.expander("📊 Mann–Kendall 检验结果", expanded=True):
                    mk_da = st.session_state.analysis_results['mk']
                    col1, col2 = st.columns(2)
                    with col1:
                        plot_map(mk_da, title="Mann–Kendall 趋势 (1=上升, -1=下降, 0=不显著)")
                    with col2:
                        st.download_button(
                            "⬇️ 下载 MK 结果 (GeoTIFF)",
                            data=dataarray_to_bytes_tif(mk_da),
                            file_name="mann_kendall.tif",
                            mime="image/tiff",
                            use_container_width=True
                        )

            # BFAST 结果
            if 'bfast' in st.session_state.analysis_results:
                with st.expander("🔍 BFAST 突变检测结果", expanded=True):
                    break_da = st.session_state.analysis_results['bfast']
                    col1, col2 = st.columns(2)
                    with col1:
                        plot_map(break_da, title="突变年份 (NaN=无突变)")
                    with col2:
                        st.download_button(
                            "⬇️ 下载突变年份 (GeoTIFF)",
                            data=dataarray_to_bytes_tif(break_da),
                            file_name="break_years.tif",
                            mime="image/tiff",
                            use_container_width=True
                        )

            # FFT 结果
            if 'fft' in st.session_state.analysis_results:
                with st.expander("📡 FFT 周期分析结果", expanded=True):
                    amp_da = st.session_state.analysis_results['fft']['amplitude']
                    period_da = st.session_state.analysis_results['fft']['period']
                    col1, col2 = st.columns(2)
                    with col1:
                        plot_map(amp_da, title="FFT 振幅")
                        st.download_button(
                            "⬇️ 下载 FFT 振幅",
                            data=dataarray_to_bytes_tif(amp_da),
                            file_name="fft_amplitude.tif",
                            mime="image/tiff",
                            use_container_width=True
                        )
                    with col2:
                        plot_map(period_da, title="FFT 主周期")
                        st.download_button(
                            "⬇️ 下载 FFT 周期",
                            data=dataarray_to_bytes_tif(period_da),
                            file_name="fft_period.tif",
                            mime="image/tiff",
                            use_container_width=True
                        )

            # STL 结果
            if 'stl' in st.session_state.analysis_results:
                with st.expander("🔄 STL 分解结果", expanded=True):
                    trend_da = st.session_state.analysis_results['stl']['trend']
                    seasonal_da = st.session_state.analysis_results['stl']['seasonal']
                    resid_da = st.session_state.analysis_results['stl']['resid']

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        plot_map(trend_da, title="STL: 平均趋势分量")
                        st.download_button(
                            "⬇️ 下载趋势分量",
                            data=dataarray_to_bytes_tif(trend_da),
                            file_name="stl_trend_mean.tif",
                            mime="image/tiff",
                            use_container_width=True
                        )
                    with col2:
                        plot_map(seasonal_da, title="STL: 平均季节分量")
                        st.download_button(
                            "⬇️ 下载季节分量",
                            data=dataarray_to_bytes_tif(seasonal_da),
                            file_name="stl_seasonal_mean.tif",
                            mime="image/tiff",
                            use_container_width=True
                        )
                    with col3:
                        plot_map(resid_da, title="STL: 残差标准差")
                        st.download_button(
                            "⬇️ 下载残差标准差",
                            data=dataarray_to_bytes_tif(resid_da),
                            file_name="stl_residual_std.tif",
                            mime="image/tiff",
                            use_container_width=True
                        )

        # ---------------------------
        # 交互式像元分析
        # ---------------------------
        st.sidebar.header("🔎 像元级分析")
        with st.expander("📈 像元时序分析工具", expanded=True):
            st.info("使用侧边栏滑杆选择特定像元查看其时序特征")

            # 根据数据频率设置默认STL周期
            pixel_stl_period = 12
            if data_frequency == "年度数据":
                pixel_stl_period = 1

            plot_pixel_timeseries(
                data_stack,
                period=st.sidebar.number_input(
                    "STL 周期参数",
                    value=pixel_stl_period,
                    min_value=1,
                    max_value=min(24, len(times) // 2),
                    key="pixel_stl_period"
                )
            )

else:
    st.info("👆 请在侧边栏上传 GeoTIFF 文件开始分析")

    # 使用说明
    with st.expander("📖 使用说明", expanded=True):
        st.markdown("""
        ### 🎯 系统功能
        - **Theil–Sen趋势分析**: 计算稳健的趋势斜率
        - **Mann–Kendall检验**: 检验趋势显著性  
        - **BFAST突变检测**: 检测时间序列中的突变点
        - **FFT周期分析**: 分析周期性特征
        - **STL分解**: 分解为趋势、季节和残差分量

        ### 📁 数据要求
        - **文件格式**: GeoTIFF (.tif, .tiff)
        - **时间信息**: 文件名必须包含时间信息
        - **年度数据命名**: `NDVI_2000.tif`, `NDVI_2001.tif`
        - **月度数据命名**: `NDVI_200001.tif`, `NDVI_200002.tif`
        - **空间范围**: 所有文件必须具有相同的空间范围和分辨率
        - **建议时间序列长度**: ≥3期以获得有意义的结果

        ### ⚡ 使用流程
        1. 在左侧上传多个GeoTIFF文件
        2. 系统自动检测数据频率（年度/月度）
        3. 选择要运行的分析方法
        4. 点击"执行选中分析"
        5. 查看结果并下载

        ### 💡 分析建议
        - **年度数据**: 适合趋势分析和突变检测
        - **月度数据**: 适合所有分析方法，特别是STL和FFT周期分析
        """)

# 页脚
st.markdown("---")
st.markdown("🛰️ 时序遥感分析系统 | 基于 Python + Streamlit 构建")
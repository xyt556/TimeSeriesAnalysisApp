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

# 页面配置
st.set_page_config(
    page_title="时序遥感分析系统_V2.0 @3S&ML",
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
    """加载并堆叠文件 - 确保保持坐标系"""
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

    # 读取数据并保持坐标系
    data_list = []
    time_coords = []
    reference_da = None

    for p, t in zip(paths, times):
        try:
            da = rxr.open_rasterio(str(p), chunks={'x': 512, 'y': 512}).squeeze()
            if "band" in da.dims:
                da = da.isel(band=0).drop_vars('band')

            # 确保数据是2D的 (y, x)
            if da.ndim != 2:
                st.error(f"文件 {p.name} 的维度不正确，期望2D数据 (y, x)，实际维度: {da.dims}")
                continue

            # 保存第一个数据作为参考
            if reference_da is None:
                reference_da = da

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
        stack = xr.concat(data_list, dim="time")
        stack = stack.assign_coords(time=time_coords)
        stack = stack.transpose('time', 'y', 'x')

        # 如果参考数据有坐标系信息，尝试应用到堆叠数据
        if reference_da is not None:
            if hasattr(reference_da, 'rio') and reference_da.rio.crs is not None:
                try:
                    stack.rio.set_crs(reference_da.rio.crs)
                except Exception as e:
                    st.warning(f"无法设置坐标系: {e}")

        # 验证数据形状和显示信息
        st.info(f"数据栈形状: {stack.shape} (时间, Y, X)")

        # 显示坐标系信息
        if hasattr(stack, 'rio') and hasattr(stack.rio, 'crs') and stack.rio.crs is not None:
            st.success(f"✅ 检测到坐标系: {stack.rio.crs}")
        else:
            st.warning("⚠️ 未检测到坐标系信息")

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
            tab1, tab2, tab3, tab4 = st.tabs(["空间分布", "时间序列抽样", "统计信息", "坐标系统"])

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

            with tab4:
                st.subheader("坐标系统信息")
                try:
                    # 检查坐标系
                    if hasattr(data_stack, 'rio') and hasattr(data_stack.rio, 'crs'):
                        crs = data_stack.rio.crs
                        if crs is not None:
                            st.success(f"✅ 坐标系: {crs}")

                            # 显示WKT格式（如果可用）
                            try:
                                st.text_area("坐标系详情 (WKT):", str(crs.to_wkt()), height=150)
                            except:
                                st.text_area("坐标系详情:", str(crs), height=100)
                        else:
                            st.warning("⚠️ 未检测到坐标系信息")
                    else:
                        st.warning("⚠️ 未检测到坐标系信息")

                    # 显示空间范围
                    try:
                        if hasattr(data_stack, 'x') and hasattr(data_stack, 'y'):
                            x_coords = data_stack.x.values
                            y_coords = data_stack.y.values
                            if len(x_coords) > 0 and len(y_coords) > 0:
                                st.write(f"X范围: {x_coords[0]:.2f} 到 {x_coords[-1]:.2f}")
                                st.write(f"Y范围: {y_coords[0]:.2f} 到 {y_coords[-1]:.2f}")
                                if len(x_coords) > 1 and len(y_coords) > 1:
                                    x_res = x_coords[1] - x_coords[0]
                                    y_res = y_coords[1] - y_coords[0]
                                    st.write(f"空间分辨率: {x_res:.6f} × {abs(y_res):.6f}")
                    except Exception as e:
                        st.write(f"无法获取空间范围: {e}")

                    # 显示变换信息
                    try:
                        if hasattr(data_stack, 'rio') and hasattr(data_stack.rio, 'transform'):
                            transform = data_stack.rio.transform()
                            if transform:
                                st.write("仿射变换参数:")
                                st.write(f"- 左上X: {transform.c}")
                                st.write(f"- 左上Y: {transform.f}")
                                st.write(f"- X分辨率: {transform.a}")
                                st.write(f"- Y分辨率: {transform.e}")
                    except Exception as e:
                        st.write(f"无法获取变换信息: {e}")

                except Exception as e:
                    st.error(f"获取坐标系信息失败: {e}")

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
                    try:
                        slope_da, intercept_da = theil_sen_trend(data_stack)
                        st.session_state.analysis_results['theilsen'] = {
                            'slope': slope_da,
                            'intercept': intercept_da
                        }
                    except Exception as e:
                        st.error(f"Theil–Sen 分析失败: {e}")
                    current_progress += 1
                    update_progress("Theil–Sen 趋势分析完成", current_progress / total_analyses)

                # Mann–Kendall 分析
                if 'mk' in selected_analyses:
                    update_progress("正在计算 Mann–Kendall 检验...", current_progress / total_analyses)
                    try:
                        mk_da = mann_kendall_test(data_stack)
                        st.session_state.analysis_results['mk'] = mk_da
                    except Exception as e:
                        st.error(f"Mann–Kendall 检验失败: {e}")
                    current_progress += 1
                    update_progress("Mann–Kendall 检验完成", current_progress / total_analyses)

                # BFAST 分析
                if 'bfast' in selected_analyses:
                    update_progress("正在检测突变点...", current_progress / total_analyses)
                    try:
                        break_da = bfast_detection(data_stack)

                        # 修复时间转换问题
                        from utils.analysis_tools import fix_bfast_results

                        break_da_fixed = fix_bfast_results(break_da)

                        st.session_state.analysis_results['bfast'] = break_da_fixed
                    except Exception as e:
                        st.error(f"BFAST 突变检测失败: {e}")
                    current_progress += 1
                    update_progress("BFAST 突变检测完成", current_progress / total_analyses)

                # FFT 分析
                if 'fft' in selected_analyses:
                    update_progress("正在进行 FFT 周期分析...", current_progress / total_analyses)
                    try:
                        amp_da, period_da = fft_analysis(data_stack)
                        st.session_state.analysis_results['fft'] = {
                            'amplitude': amp_da,
                            'period': period_da
                        }
                    except Exception as e:
                        st.error(f"FFT 周期分析失败: {e}")
                    current_progress += 1
                    update_progress("FFT 周期分析完成", current_progress / total_analyses)

                # STL 分解
                if 'stl' in selected_analyses:
                    if data_frequency == "年度数据":
                        st.warning("⚠️ 年度数据不适合STL分解，季节周期可能无意义")

                    update_progress("正在执行 STL 分解...", current_progress / total_analyses)
                    try:
                        trend_da, seasonal_da, resid_da = stl_decompose_pixelwise(
                            data_stack,
                            period=stl_period
                        )
                        st.session_state.analysis_results['stl'] = {
                            'trend': trend_da,
                            'seasonal': seasonal_da,
                            'resid': resid_da
                        }
                    except Exception as e:
                        st.error(f"STL 分解失败: {e}")
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
                            use_container_width=True,
                            help="下载的TIFF文件将保持原始坐标系和空间参考信息"
                        )

            # Mann–Kendall 结果
            if 'mk' in st.session_state.analysis_results:
                with st.expander("📊 Mann–Kendall 检验结果", expanded=True):
                    mk_da = st.session_state.analysis_results['mk']
                    col1, col2 = st.columns(2)
                    with col1:
                        # 使用专门的MK绘图参数
                        plot_map(mk_da, title="Mann–Kendall 趋势 (1=上升, -1=下降, 0=不显著)",
                                 vmin=-1, vmax=1)

                        # 显示统计信息
                        mk_values = mk_da.values
                        valid_mask = ~np.isnan(mk_values)
                        if np.any(valid_mask):
                            valid_values = mk_values[valid_mask]
                            st.write("趋势统计:")
                            st.write(f"- 显著上升: {np.sum(valid_values == 1)} 像元")
                            st.write(f"- 显著下降: {np.sum(valid_values == -1)} 像元")
                            st.write(f"- 无显著趋势: {np.sum(valid_values == 0)} 像元")

                    with col2:
                        st.download_button(
                            "⬇️ 下载 MK 结果 (GeoTIFF)",
                            data=dataarray_to_bytes_tif(mk_da),
                            file_name="mann_kendall.tif",
                            mime="image/tiff",
                            use_container_width=True,
                            help="下载的TIFF文件将保持原始坐标系和空间参考信息"
                        )

            # BFAST 结果
            if 'bfast' in st.session_state.analysis_results:
                with st.expander("🔍 BFAST 突变检测结果", expanded=True):
                    break_da = st.session_state.analysis_results['bfast']

                    # 添加结果解释
                    st.info("""
                    **结果解读指南：**
                    - **数值**: 检测到的突变发生年份（如 2020, 2021 等）
                    - **NaN**: 无显著突变
                    - **突变含义**: 时间序列中发生显著变化的时刻，可能对应自然灾害、人类活动、政策变化等事件
                    """)

                    col1, col2 = st.columns(2)
                    with col1:
                        # 对BFAST结果进行特殊处理，确保显示正确的年份
                        break_values = break_da.values

                        # 过滤掉异常大的值（可能是未转换的时间戳）
                        if np.nanmax(break_values) > 3000:  # 如果最大值超过3000，说明可能是时间戳
                            st.warning("⚠️ 检测到异常时间值，正在进行转换...")
                            # 转换时间戳到年份
                            break_values_converted = np.full_like(break_values, np.nan)
                            for i in range(break_values.shape[0]):
                                for j in range(break_values.shape[1]):
                                    val = break_values[i, j]
                                    if not np.isnan(val):
                                        if val > 1000000000000000000:  # 可能是纳秒时间戳
                                            # 转换为年份
                                            try:
                                                dt = pd.to_datetime(val)
                                                break_values_converted[i, j] = dt.year
                                            except:
                                                break_values_converted[i, j] = np.nan
                                        elif val > 3000:  # 异常大的年份值
                                            break_values_converted[i, j] = np.nan
                                        else:
                                            break_values_converted[i, j] = val

                            # 创建新的DataArray
                            break_da_corrected = xr.DataArray(
                                break_values_converted,
                                dims=break_da.dims,
                                coords=break_da.coords
                            )

                            # 更新session state中的结果
                            st.session_state.analysis_results['bfast'] = break_da_corrected
                            break_da = break_da_corrected

                        plot_map(break_da, title="突变年份 (NaN=无突变)")

                        # 显示统计信息
                        break_values = break_da.values
                        valid_mask = ~np.isnan(break_values)
                        if np.any(valid_mask):
                            valid_years = break_values[valid_mask]

                            # 过滤异常年份（只保留合理的年份范围）
                            current_year = datetime.datetime.now().year
                            valid_years_filtered = valid_years[
                                (valid_years >= 1900) & (valid_years <= current_year + 1)
                                ]

                            st.write("突变统计:")
                            st.write(f"- 检测到突变的像元: {len(valid_years_filtered)} 个")

                            # 显示突变年份分布
                            if len(valid_years_filtered) > 0:
                                unique_years, counts = np.unique(valid_years_filtered.astype(int), return_counts=True)
                                st.write("突变年份分布:")
                                for year, count in zip(unique_years, counts):
                                    st.write(f"  - {year}年: {count} 像元")

                                # 绘制年份分布图
                                fig, ax = plt.subplots(figsize=(10, 4))
                                ax.bar(unique_years, counts, color='skyblue', alpha=0.7)
                                ax.set_xlabel('年份')
                                ax.set_ylabel('像元数量')
                                ax.set_title('突变年份分布')
                                ax.grid(True, alpha=0.3)
                                plt.xticks(rotation=45)
                                plt.tight_layout()
                                st.pyplot(fig)
                                plt.close(fig)

                    with col2:
                        st.download_button(
                            "⬇️ 下载突变年份 (GeoTIFF)",
                            data=dataarray_to_bytes_tif(break_da),
                            file_name="break_years.tif",
                            mime="image/tiff",
                            use_container_width=True,
                            help="下载的TIFF文件中，像元值代表突变发生的年份，NaN表示无突变"
                        )

                        # 添加数据说明
                        st.markdown("""
                        **数据说明：**
                        - 像元值：突变发生的具体年份
                        - 数据范围：1900-当前年份
                        - 异常值已自动过滤
                        """)

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
                            use_container_width=True,
                            help="下载的TIFF文件将保持原始坐标系和空间参考信息"
                        )
                    with col2:
                        plot_map(period_da, title="FFT 主周期")
                        st.download_button(
                            "⬇️ 下载 FFT 周期",
                            data=dataarray_to_bytes_tif(period_da),
                            file_name="fft_period.tif",
                            mime="image/tiff",
                            use_container_width=True,
                            help="下载的TIFF文件将保持原始坐标系和空间参考信息"
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
                            use_container_width=True,
                            help="下载的TIFF文件将保持原始坐标系和空间参考信息"
                        )
                    with col2:
                        plot_map(seasonal_da, title="STL: 平均季节分量")
                        st.download_button(
                            "⬇️ 下载季节分量",
                            data=dataarray_to_bytes_tif(seasonal_da),
                            file_name="stl_seasonal_mean.tif",
                            mime="image/tiff",
                            use_container_width=True,
                            help="下载的TIFF文件将保持原始坐标系和空间参考信息"
                        )
                    with col3:
                        plot_map(resid_da, title="STL: 残差标准差")
                        st.download_button(
                            "⬇️ 下载残差标准差",
                            data=dataarray_to_bytes_tif(resid_da),
                            file_name="stl_residual_std.tif",
                            mime="image/tiff",
                            use_container_width=True,
                            help="下载的TIFF文件将保持原始坐标系和空间参考信息"
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
        - **月度数据命名**: `NDVI_200001.tif`, `NDVI_2000_01.tif`, `NDVI_2000_01_徐州.tif`
        - **日度数据命名**: `NDVI_2000_001.tif`, `NDVI_2000_365_徐州.tif`
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

        ### ❗ 常见问题
        - **一个时间点多个值**: 确保文件名中的时间信息唯一，不要有重复的时间点
        - **数据维度错误**: 确保所有文件都是单波段2D栅格数据
        - **时间坐标错误**: 检查文件名中的时间格式是否正确
        - **坐标系问题**: 系统会自动保持原始文件的坐标系信息
        """)

# 页脚
st.markdown("---")
st.markdown("🛰️ 时序遥感分析系统 | 基于 Python + Streamlit 构建")
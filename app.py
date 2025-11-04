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

warnings.filterwarnings('ignore')

# # 设置中文字体
# matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'STHeiti']
# matplotlib.rcParams['axes.unicode_minus'] = False



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

st.title("🛰️ 时序遥感分析系统--@3S&ML")
st.markdown("""
**功能模块：** Theil–Sen趋势分析 | Mann–Kendall检验 | BFAST突变检测 | FFT周期分析 | STL分解
""")


# ---------------------------
# 文件上传与预处理
# ---------------------------
@st.cache_data(show_spinner=False)
def extract_year(filename):
    """提取文件名中的年份"""
    m = re.search(r'(19|20)\d{2}', filename)
    return int(m.group(0)) if m else None


@st.cache_data(show_spinner=False)
def load_and_stack_files(_uploaded_files, years):
    """加载并堆叠文件"""
    tmpdir = Path(tempfile.mkdtemp())
    paths = []

    for f in _uploaded_files:
        p = tmpdir / f.name
        p.write_bytes(f.getbuffer())
        paths.append(p)

    # 并行读取文件
    data_list = []
    for p in paths:
        try:
            da = rxr.open_rasterio(str(p), chunks={'x': 512, 'y': 512}).squeeze()
            if "band" in da.dims:
                da = da.isel(band=0).drop_vars('band')
            data_list.append(da)
        except Exception as e:
            st.error(f"读取文件 {p.name} 时出错: {e}")
            continue

    if not data_list:
        st.error("没有成功读取任何文件")
        return None

    # 堆叠数据
    try:
        stack = xr.concat(data_list, dim="time")
        stack = stack.assign_coords(time=years)
        # 确保坐标一致性
        stack = stack.transpose('time', 'y', 'x')
        return stack
    except Exception as e:
        st.error(f"数据堆叠失败: {e}")
        return None


# 文件上传区域
st.sidebar.header("📁 数据上传")
uploaded_files = st.sidebar.file_uploader(
    "上传多期 GeoTIFF 文件",
    type=["tif", "tiff"],
    accept_multiple_files=True,
    help="文件名应包含年份，如: NDVI_2000.tif 或 any1999.tif"
)

# 初始化session state
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = {}
if 'data_stack' not in st.session_state:
    st.session_state.data_stack = None

if uploaded_files:
    # 按文件名排序以确保一致性
    uploaded_files = sorted(uploaded_files, key=lambda f: f.name)
    years = [extract_year(f.name) for f in uploaded_files]

    # 检查年份提取
    invalid_files = [(f.name, y) for f, y in zip(uploaded_files, years) if y is None]
    if invalid_files:
        st.error("以下文件中未检测到有效年份:")
        for fname, year in invalid_files:
            st.error(f"  - {fname}")
        st.info("💡 请确保文件名包含4位年份，如: 2000, 1999")
    else:
        st.success(f"✅ 成功检测到 {len(uploaded_files)} 期数据，年份: {sorted(years)}")

        # 加载数据
        with st.spinner("🔄 正在加载和处理栅格数据..."):
            data_stack = load_and_stack_files(uploaded_files, years)

        if data_stack is not None:
            st.session_state.data_stack = data_stack

            # 显示数据信息
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("时间序列长度", f"{data_stack.sizes['time']} 期")
            with col2:
                st.metric("空间分辨率", f"{data_stack.sizes['y']} × {data_stack.sizes['x']}")
            with col3:
                st.metric("年份范围", f"{min(years)} - {max(years)}")

            # 数据预览
            with st.expander("📊 数据预览", expanded=True):
                tab1, tab2 = st.tabs(["空间分布", "统计信息"])
                with tab1:
                    st.subheader("第一期栅格预览")
                    plot_map(data_stack.isel(time=0),
                             title=f"年份 {data_stack.time.values[0]}")
                with tab2:
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

            # STL参数设置
            if 'stl' in selected_analyses:
                stl_period = st.sidebar.number_input(
                    "STL 周期参数",
                    value=min(12, len(years) // 2),
                    min_value=2,
                    max_value=len(years) // 2,
                    help="季节周期长度，通常为12(月数据)或其他"
                )

            # 执行按钮
            run_analysis = st.sidebar.button(
                "🚀 执行选中分析",
                type="primary",
                use_container_width=True
            )

            if run_analysis and selected_analyses:
                progress_bar = st.progress(0)
                status_text = st.empty()

                try:
                    # Theil–Sen 分析
                    if 'theilsen' in selected_analyses:
                        status_text.text("📈 正在计算 Theil–Sen 趋势...")
                        slope_da, intercept_da = theil_sen_trend(data_stack)
                        st.session_state.analysis_results['theilsen'] = {
                            'slope': slope_da,
                            'intercept': intercept_da
                        }
                        progress_bar.progress(20)

                    # Mann–Kendall 分析
                    if 'mk' in selected_analyses:
                        status_text.text("📊 正在计算 Mann–Kendall 检验...")
                        mk_da = mann_kendall_test(data_stack)
                        st.session_state.analysis_results['mk'] = mk_da
                        progress_bar.progress(40)

                    # BFAST 分析
                    if 'bfast' in selected_analyses:
                        status_text.text("🔍 正在检测突变点...")
                        break_da = bfast_detection(data_stack)
                        st.session_state.analysis_results['bfast'] = break_da
                        progress_bar.progress(60)

                    # FFT 分析
                    if 'fft' in selected_analyses:
                        status_text.text("📡 正在进行 FFT 周期分析...")
                        amp_da, period_da = fft_analysis(data_stack)
                        st.session_state.analysis_results['fft'] = {
                            'amplitude': amp_da,
                            'period': period_da
                        }
                        progress_bar.progress(80)

                    # STL 分解
                    if 'stl' in selected_analyses:
                        status_text.text("🔄 正在执行 STL 分解...")
                        trend_da, seasonal_da, resid_da = stl_decompose_pixelwise(
                            data_stack,
                            period=stl_period
                        )
                        st.session_state.analysis_results['stl'] = {
                            'trend': trend_da,
                            'seasonal': seasonal_da,
                            'resid': resid_da
                        }
                        progress_bar.progress(100)

                    status_text.text("✅ 所有分析完成!")
                    st.balloons()

                except Exception as e:
                    st.error(f"分析过程中出错: {e}")
                    progress_bar.empty()
                    status_text.empty()

                finally:
                    progress_bar.empty()
                    status_text.empty()

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
                            # 注意：trend_da 已经是平均值，不需要再调用 .mean("time")
                            plot_map(trend_da, title="STL: 平均趋势分量")
                            st.download_button(
                                "⬇️ 下载趋势分量",
                                data=dataarray_to_bytes_tif(trend_da),
                                file_name="stl_trend_mean.tif",
                                mime="image/tiff",
                                use_container_width=True
                            )
                        with col2:
                            # seasonal_da 也已经是平均值
                            plot_map(seasonal_da, title="STL: 平均季节分量")
                            st.download_button(
                                "⬇️ 下载季节分量",
                                data=dataarray_to_bytes_tif(seasonal_da),
                                file_name="stl_seasonal_mean.tif",
                                mime="image/tiff",
                                use_container_width=True
                            )
                        with col3:
                            # resid_da 已经是标准差
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
                plot_pixel_timeseries(
                    data_stack,
                    period=st.sidebar.number_input(
                        "STL 周期参数",
                        value=min(12, len(years) // 2),
                        min_value=2,
                        max_value=len(years) // 2,
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
        - 文件格式: GeoTIFF (.tif, .tiff)
        - 文件名: 必须包含4位年份，如 `NDVI_2000.tif`
        - 空间范围: 所有文件必须具有相同的空间范围和分辨率
        - 建议时间序列长度: ≥3期以获得有意义的结果

        ### ⚡ 使用流程
        1. 在左侧上传多个GeoTIFF文件
        2. 选择要运行的分析方法
        3. 点击"执行选中分析"
        4. 查看结果并下载
        """)

# 页脚
st.markdown("---")
st.markdown("🛰️ 时序遥感分析系统 | 基于 Python + Streamlit 构建")
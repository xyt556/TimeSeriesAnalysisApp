# utils/visualization.py
import io
import numpy as np
import rasterio
from rasterio.io import MemoryFile
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import streamlit as st
import pandas as pd

# 将 xarray DataArray -> 2D numpy（若含 time 维则 collapse time mean）
def _da_to_2d(da):
    try:
        import xarray as xr
        if "time" in da.dims and "y" in da.dims and "x" in da.dims:
            return np.nanmean(da.values, axis=0)
        elif "y" in da.dims and "x" in da.dims:
            return da.values
        else:
            vals = da.values
            if vals.ndim >= 2:
                return np.nanmean(vals, axis=0)
            return vals
    except Exception:
        return np.array(da)

def plot_map(da, title="", cmap="RdBu_r", vmin=None, vmax=None, fig_width=6, fig_height=5):
    img = _da_to_2d(da)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    if np.nanmin(img) < 0 and np.nanmax(img) > 0:
        norm = TwoSlopeNorm(vcenter=0, vmin=(vmin if vmin is not None else np.nanmin(img)),
                            vmax=(vmax if vmax is not None else np.nanmax(img)))
        im = ax.imshow(img, cmap=cmap, norm=norm)
    else:
        im = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title(title)
    ax.axis("off")
    st.pyplot(fig)
    return fig

def plot_pixel_timeseries(stack, row=None, col=None, period=12):
    # stack: xarray.DataArray (time,y,x)
    ny = int(stack.sizes["y"])
    nx = int(stack.sizes["x"])
    if row is None:
        row = st.sidebar.slider("row", 0, ny-1, ny//2)
    if col is None:
        col = st.sidebar.slider("col", 0, nx-1, nx//2)

    series = stack[:, row, col].values
    times = stack["time"].values

    fig, axs = plt.subplots(3, 1, figsize=(8, 6), sharex=True)
    axs[0].plot(times, series, marker="o"); axs[0].set_title(f"Pixel ({row},{col}) Time Series")
    axs[0].grid(True)

    # STL
    try:
        from statsmodels.tsa.seasonal import STL
        if np.count_nonzero(~np.isnan(series)) >= max(3, period*2):
            stl = STL(series, period=period, robust=True).fit()
            axs[1].plot(times, stl.trend); axs[1].set_title("STL Trend")
            axs[2].plot(times, stl.seasonal); axs[2].set_title("STL Seasonal")
        else:
            axs[1].text(0.1, 0.5, "时间序列太短，无法 STL 分解", transform=axs[1].transAxes)
            axs[2].axis("off")
    except Exception as e:
        axs[1].text(0.1, 0.5, f"STL 失败: {e}", transform=axs[1].transAxes)
        axs[2].axis("off")

    axs[2].set_xlabel("time")
    plt.tight_layout()
    st.pyplot(fig)

    # 下载 PNG
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=150); buf.seek(0)
    st.download_button("⬇️ 下载像元时序图 (PNG)", data=buf.getvalue(), file_name=f"pixel_{row}_{col}.png", mime="image/png")
    buf.close()

    # 下载 CSV
    df = pd.DataFrame({"time": times, "value": series})
    csvb = df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ 下载像元时序 (CSV)", data=csvb, file_name=f"pixel_{row}_{col}.csv", mime="text/csv")
    return fig

def dataarray_to_bytes_tif(da):
    # produce a 2D GeoTIFF bytes; try to use da.rio.profile if available
    arr2d = _da_to_2d(da)
    if hasattr(da, "rio") and hasattr(da.rio, "transform"):
        try:
            profile = da.rio.profile
            # update profile
            profile.update(dtype=rasterio.float32, count=1, compress="lzw")
        except Exception:
            profile = None
    else:
        profile = None

    # fallback minimal profile
    import rasterio
    from rasterio.io import MemoryFile
    if profile is None:
        profile = {
            "driver": "GTiff",
            "dtype": rasterio.float32,
            "count": 1,
            "height": arr2d.shape[0],
            "width": arr2d.shape[1],
            "transform": rasterio.transform.from_origin(0, 0, 1, 1),
            "crs": None,
            "compress": "lzw"
        }

    arr2d = np.nan_to_num(arr2d, nan=-9999.0).astype(np.float32)
    memfile = MemoryFile()
    with memfile.open(**profile) as dst:
        dst.write(arr2d, 1)
    data = memfile.read()
    memfile.close()
    return data

def fig_to_bytes_png(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
    buf.seek(0)
    data = buf.read()
    buf.close()
    return data

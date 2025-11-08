# core/clustering.py
"""
时间序列聚类分析模块
"""

import numpy as np
import xarray as xr
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score

from config import Config
from utils import logger


class TimeSeriesClusterer:
    """时间序列聚类分析器"""

    @staticmethod
    def kmeans_clustering(stack: xr.DataArray, n_clusters=None,
                          max_iter=100, random_state=42,
                          progress_tracker=None):
        """K-means聚类

        Args:
            stack: 时间序列数据
            n_clusters: 聚类数量
            max_iter: 最大迭代次数
            random_state: 随机种子
            progress_tracker: 进度跟踪器

        Returns:
            (cluster_map, centers, metrics): 聚类地图、中心、质量指标
        """
        if n_clusters is None:
            n_clusters = Config.CLUSTER_DEFAULT

        logger.info(f"Starting K-means clustering (n_clusters={n_clusters})")

        data = stack.values
        n_time, ny, nx = data.shape

        if progress_tracker:
            progress_tracker.update("准备聚类数据", 5)

        # 重塑为 (n_pixels, n_time)
        reshaped = data.transpose(1, 2, 0).reshape(-1, n_time)

        # 移除全NaN的像元
        valid_mask = ~np.all(np.isnan(reshaped), axis=1)
        valid_data = reshaped[valid_mask]

        logger.info(f"Valid pixels for clustering: {len(valid_data)}")

        if progress_tracker:
            progress_tracker.update("数据插值", 15)

        # 对有NaN的序列进行插值
        for i in range(len(valid_data)):
            ts = valid_data[i]
            if np.any(np.isnan(ts)):
                mask = ~np.isnan(ts)
                if np.sum(mask) >= 2:
                    x = np.arange(n_time)
                    valid_data[i] = np.interp(x, x[mask], ts[mask])
                else:
                    valid_data[i] = np.nanmean(ts)

        if progress_tracker:
            progress_tracker.update("数据标准化", 25)

        # 标准化
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(valid_data)

        if progress_tracker:
            progress_tracker.update("K-means聚类计算", 35)

        # 聚类
        kmeans = KMeans(
            n_clusters=n_clusters,
            max_iter=max_iter,
            random_state=random_state,
            n_init=10
        )
        labels = kmeans.fit_predict(scaled_data)

        if progress_tracker:
            progress_tracker.update("生成结果", 80)

        # 重建为空间形状
        full_labels = np.full(ny * nx, -1, dtype=int)
        full_labels[valid_mask] = labels
        cluster_map = full_labels.reshape(ny, nx)

        # 计算聚类中心（原始尺度）
        centers = scaler.inverse_transform(kmeans.cluster_centers_)

        # 计算聚类质量指标
        try:
            metrics = {
                'inertia': float(kmeans.inertia_),
                'silhouette': float(silhouette_score(scaled_data, labels)),
                'calinski_harabasz': float(calinski_harabasz_score(scaled_data, labels))
            }
        except:
            metrics = {'inertia': float(kmeans.inertia_)}

        result = xr.DataArray(
            cluster_map,
            dims=('y', 'x'),
            coords={'y': stack.y, 'x': stack.x}
        )

        logger.info(f"Clustering completed. Metrics: {metrics}")

        if progress_tracker:
            progress_tracker.update("聚类完成", 100)

        return result, centers, metrics

    @staticmethod
    def hierarchical_clustering(stack: xr.DataArray, n_clusters=None,
                                linkage='ward', progress_tracker=None):
        """层次聚类

        Args:
            stack: 时间序列数据
            n_clusters: 聚类数量
            linkage: 链接方法
            progress_tracker: 进度跟踪器

        Returns:
            cluster_map: 聚类地图
        """
        if n_clusters is None:
            n_clusters = Config.CLUSTER_DEFAULT

        logger.info(f"Starting hierarchical clustering (n_clusters={n_clusters}, linkage={linkage})")

        data = stack.values
        n_time, ny, nx = data.shape

        reshaped = data.transpose(1, 2, 0).reshape(-1, n_time)
        valid_mask = ~np.all(np.isnan(reshaped), axis=1)
        valid_data = reshaped[valid_mask]

        # 插值处理NaN
        for i in range(len(valid_data)):
            ts = valid_data[i]
            if np.any(np.isnan(ts)):
                mask = ~np.isnan(ts)
                if np.sum(mask) >= 2:
                    x = np.arange(n_time)
                    valid_data[i] = np.interp(x, x[mask], ts[mask])

        # 标准化和聚类
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(valid_data)

        if progress_tracker:
            progress_tracker.update("层次聚类计算", 50)

        clustering = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage=linkage
        )
        labels = clustering.fit_predict(scaled_data)

        # 重建空间形状
        full_labels = np.full(ny * nx, -1, dtype=int)
        full_labels[valid_mask] = labels
        cluster_map = full_labels.reshape(ny, nx)

        result = xr.DataArray(
            cluster_map,
            dims=('y', 'x'),
            coords={'y': stack.y, 'x': stack.x}
        )

        logger.info("Hierarchical clustering completed")
        return result
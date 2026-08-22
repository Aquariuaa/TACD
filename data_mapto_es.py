import pandas as pd
import numpy as np

# data simulation, simulate true causal dag and train_data.
def create_multi_es(X, delta):
    # print("X.shape", X.shape)
    # print("type(X)", type(X))
    # 使用pivot_table将event类型转换为列
    window_size = delta
    if window_size == 1:
        multi_series = X.pivot_table(
            index='timestamp',
            columns='event',
            values='node',
            aggfunc='count'  # 统计每个时间戳下每个event出现的次数
        ).fillna(0)
    else:
        # 创建时间窗口
        X['time_window'] = (X['timestamp'] // window_size) * window_size
        # 统计每个时间窗口内的事件数量
        multi_series = X.pivot_table(
            index='time_window',
            columns='event',
            values='node',
            aggfunc='count'
        ).fillna(0)

    # print("转换后的形状:", multi_series.shape)
    # print("\n前5行:")
    # print(multi_series.head())
    return multi_series


def downsample_by_time_interval(data, d):
    """
    对事件序列数据按时间戳排序后进行间隔采样

    参数:
    - data: DataFrame, 需要下采样的原始数据，需包含 'timestamp' 列
    - d: int, 采样间隔（每隔d个时间戳取一个样本）

    返回:
    - DataFrame: 下采样后的数据，保持原始数据结构
    """
    if d <= 0:
        raise ValueError("采样间隔d必须为正整数")

    # 1. 按时间戳排序
    sorted_data = data.sort_values('timestamp').reset_index(drop=True)

    # 2. 计算采样索引
    n_samples = len(sorted_data)

    # 如果d大于数据量，则只取第一个样本
    if d >= n_samples:
        sampled_indices = [0]
    else:
        # 创建从0开始的索引数组，步长为d
        sampled_indices = np.arange(0, n_samples, d)

    # 3. 按索引取样
    sampled_data = sorted_data.iloc[sampled_indices].copy()

    # 4. 重置索引保持整洁
    sampled_data.reset_index(drop=True, inplace=True)

    return sampled_data


# # 使用示例
# # 假设你的数据是 X，采样间隔 d=10
# X2 = downsample_by_time_interval(X, d=2)
#
# # 打印原始数据和采样后数据的对比信息
# print("原始数据 X:")
# print(f"  数据形状: {X.shape}")
# print(f"  时间戳范围: {X['timestamp'].min()} 到 {X['timestamp'].max()}")
# print(f"  事件数量: {len(X)}")
# print()
#
# print("采样后数据 X2:")
# print(f"  数据形状: {X2.shape}")
# print(f"  时间戳范围: {X2['timestamp'].min()} 到 {X2['timestamp'].max()}")
# print(f"  事件数量: {len(X2)}")
# print(f"  采样率: 约 {len(X2) / len(X) * 100:.2f}% (1/{len(X) // len(X2)})")
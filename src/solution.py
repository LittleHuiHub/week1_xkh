# import pandas as pd
# import numpy as np
#
#
# def ops_rolling_rank(input_path: str, window: int = 20) -> np.ndarray:
#     df = pd.read_parquet(input_path)
#
#     def rolling_percentile_rank(series):
#         def rank_window(window_data):
#             if len(window_data) == 0:
#                 return np.nan
#             current_val = window_data.iloc[-1]
#             rank_sum = (window_data <= current_val).sum()
#             return rank_sum / len(window_data)
#
#         return series.rolling(window=window, min_periods=1).apply(
#             rank_window, raw=False
#         )
#
#     ranks = df.groupby('symbol', group_keys=False)['Close'].apply(
#         rolling_percentile_rank
#     )
#
#     res = ranks.values.astype(np.float32)
#     return res[:, None] # must be [N, 1]
#
#



import pandas as pd
import numpy as np


def ops_rolling_rank(input_path: str, window: int = 20) -> np.ndarray:
    """
    高性能滚动百分位秩计算：
    对每个 symbol 的 Close 序列，计算每个位置 i 上：
        rank[i] = count({ j in [i-window+1, i] | Close[j] <= Close[i] }) / window_len
    其中 window_len = i - start + 1，且 start = max(0, i-window+1)。
    返回形状为 [N, 1] 的 float32 数组，与参考答案严格对齐（含 NaN 对齐）。
    """
    df = pd.read_parquet(input_path)

    # 保持原有顺序输出，与参考答案按行对齐（假设输入已按时间排序）。
    # 需要对每个 symbol 分别计算，再写回对应位置。
    n = len(df)
    out = np.empty(n, dtype=np.float32)

    # 取基础列
    symbols = df["symbol"].to_numpy()
    closes = df["Close"].to_numpy()

    # 为了按原行顺序回填，先分组得到每组的索引切片
    # 使用 pandas 分组的 indices 能拿到每个 symbol 的整型位置索引列表
    gb = df.groupby("symbol", sort=False, observed=True)
    groups = gb.indices  # dict: symbol -> ndarray of positions (in original order)

    # 对每个 symbol 单独处理
    for _, idx in groups.items():
        # 该组的 Close 子数组视图
        x = closes[idx]

        # 单调非降双端队列，存储的是索引（相对于 x 的局部索引）
        # 队尾保持较大的值，队首是窗口内最小值的索引
        # 但我们需要的是 <= x[i] 的个数。
        # 技巧：用“单调非降队列”存储候选索引，当移动窗口和加入新元素时，
        # 将所有大于当前值的元素从队尾弹出，只保留 <= 当前值的索引。
        # 队列长度即是 count(x_j <= x_i)（仅统计当前窗口范围内的），
        # 再除以窗口内元素个数得到百分比。
        from collections import deque

        dq = deque()  # 存储局部索引 j，保证 x[dq[0]] <= x[dq[1]] <= ... <= x[dq[-1]]

        # 窗口左边界（局部）：left = max(0, i-window+1)
        left = 0
        for i_local in range(len(x)):
            # 更新窗口左边界
            left = i_local - window + 1
            if left < 0:
                left = 0

            # 弹出不在窗口范围内的索引（从队首）
            while dq and dq[0] < left:
                dq.popleft()

            # 维护单调非降：弹出所有 > 当前值 的索引（保留等于当前值的，保证 <= 计数包含相等）
            xi = x[i_local]
            while dq and x[dq[-1]] > xi:
                dq.pop()

            # 将当前索引加入队尾
            dq.append(i_local)

            # 此时 dq 中的所有索引都满足在窗口内且其值 <= 当前值
            count_le = len(dq)

            # 窗口长度
            window_len = i_local - left + 1

            # 百分位秩
            out[idx[i_local]] = np.float32(count_le / window_len)

    # 返回 [N, 1]
    return out[:, None]
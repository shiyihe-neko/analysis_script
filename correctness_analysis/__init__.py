import pandas as pd
import re
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st
from scipy.stats import pearsonr


def _normalize(val: any) -> str:
    """
    基础规范化：
      - 转成 str
      - 折叠空白
      - strip
      - lower
    """
    s = str(val)
    s = re.sub(r"\s+", " ", s).strip()
    return s.lower()


def annotate_reading_correctness_exact(
    df: pd.DataFrame,
    correct_answers: dict,
    task_col: str     = 'task',
    format_col: str   = 'format',
    response_col: str = 'response'
) -> pd.DataFrame:
    """
    精确匹配阅读题答案：
    - correct_answers[key] 为单值或列表，值可以是类似 '[Bob, Eve]' 或 'Alice'
    - 只当用户作答完全等于某一选项（单选或多选整体）时，才算正确

    对于用户 response：
      - 若为 list/tuple，则视为多选，multiple_norm = ','.join(norm(item))
      - 若单值或字符串，则单选，item_norm = norm(item)
    将正确答案列表中的每一项：
      - 把外层方括号去掉得到 inner = strip_brackets(ans)
      - norm(inner) 作为候选 multi_norm
      - norm(ans) 作为候选 single_norm

    匹配规则：
      - 如果是多选（len>1），比较 multi_norm 是否在候选 multi_norms
      - 如果是单选，则比较 item_norm 是否在候选 single_norms

    返回含新增两列的 DataFrame：
      - correct_answer: 原始候选拼接字符串
      - correct: 0/1
    """
    df2 = df.copy()
    ans_col = []
    flag_col = []

    for _, row in df2.iterrows():
        # 清理 key
        orig_task = str(row[task_col])
        fmt = str(row[format_col])
        clean_key = re.sub(fr"-{re.escape(fmt)}(?=-\d+$)", "", orig_task)
        raw = correct_answers.get(clean_key, correct_answers.get(orig_task, []))
        if not isinstance(raw, (list, tuple)):
            raw = [raw]
        # display
        ans_col.append(", ".join(str(x) for x in raw)
        )
        # build candidate norms
        single_norms = set()
        multi_norms  = set()
        for cand in raw:
            cand_str = str(cand)
            # inner = remove outer brackets
            inner = re.sub(r"^\s*\[|\]\s*$", "", cand_str)
            # normalize
            sn = _normalize(inner)
            single_norms.add(sn)
            # multi_norm uses commas
            parts = [_normalize(x) for x in inner.split(',')]
            mn = ",".join(parts)
            multi_norms.add(mn)

        # process response
        resp = row[response_col]
        # if list/tuple -> multi, else single
        if isinstance(resp, (list, tuple)):
            parts = [_normalize(x) for x in resp]
            resp_norm = ",".join(parts)
            hit = resp_norm in multi_norms
        else:
            r = str(resp)
            # remove brackets if any
            r_inner = re.sub(r"^\s*\[|\]\s*$", "", r)
            r_norm = _normalize(r_inner)
            hit = r_norm in single_norms
        flag_col.append(int(hit))

    df2['correct_answer'] = ans_col
    df2['correct'] = flag_col
    return df2


def visualize_accuracy_heatmap(
    df: pd.DataFrame,
    task_col: str = 'task',
    format_col: str = 'format',
    accuracy_col: str = 'correctness',
    task_list: list = None,
    format_list: list = None,
    cmap: str = 'Reds',
    annot: bool = True,
    fmt: str = '.2f',
    cbar_label: str = 'Accuracy'
):
    """
    画一个“任务 × 格式” 的正确率热力图，并在右侧和底部各加一列/行平均值。
    """
    df2 = df.copy()

    # —— 可选过滤 —— 
    if task_list is not None:
        df2 = df2[df2[task_col].isin(task_list)]
    if format_list is not None:
        df2 = df2[df2[format_col].isin(format_list)]
    if df2.empty:
        raise ValueError("过滤后没有数据: 请检查 task_list / format_list 是否正确。")

    # —— 计算 pivot（平均正确率） —— 
    pivot = (
        df2
        .groupby([task_col, format_col])[accuracy_col]
        .mean()
        .unstack(fill_value=0)
    )

    # —— 增加右侧“Average”列 —— 
    pivot_avg = pivot.copy()
    pivot_avg['Average'] = pivot.mean(axis=1)

    # —— 增加底部“Average”行 —— 
    # 先计算各列（包括新加的 Average 列）的平均
    avg_row = pivot_avg.mean(axis=0)
    # 将其追加为索引为 'Average' 的一行
    pivot_avg.loc['Average'] = avg_row

    # —— 调整图尺寸，考虑多了一行一列 —— 
    n_tasks   = pivot_avg.shape[0]  # 原来 + 1
    n_formats = pivot_avg.shape[1]  # 原来 + 1
    figsize = (max(6, n_formats * 1.2), max(4, n_tasks * 0.6))

    # —— 绘制 heatmap —— 
    plt.figure(figsize=figsize)
    ax = sns.heatmap(
        pivot_avg,
        cmap=cmap,
        annot=annot,
        fmt=fmt,
        cbar_kws={'label': cbar_label},
        linewidths=.5,
        linecolor='gray'
    )

    ax.set_xlabel(format_col)
    ax.set_ylabel(task_col)
    ax.set_title("Accuracy Heatmap: Task vs Format\n(with row/column averages)")
    plt.tight_layout()
    plt.show()



def visualize_score_distribution_heatmap(
    df: pd.DataFrame,
    participant_col: str = 'participantId',
    task_col: str = 'task',
    format_col: str = 'format',
    correct_col: str = 'correct',
    max_score: int = None,
    cmap: str = 'Blues',
    annot: bool = True,
    fmt: str = 'd',
    cbar_label: str = 'Number of Participants',
    total_color=None  # 可以是单色名、colormap 名称或颜色列表
):
    """
    在“格式×答对题数”热力图中，右侧和下侧 Totals 区域使用渐变色显示。
    
    total_color:
      - None（默认）：使用 light gray 渐变。
      - str，且是 matplotlib 内置 colormap 名称，如 'Greys','Purples'。
      - str，且是单一颜色，如 'lightgray'，会做白→该色渐变。
      - list of colors：渐变过渡此列表中指定的颜色。
    """
    # --- 数据预处理（同之前） ---
    df2 = df.copy()
    if 'task_all' in df2[task_col].unique():
        df_counts = (
            df2[df2[task_col] == 'task_all']
            [[participant_col, format_col, correct_col]]
            .rename(columns={correct_col: 'correct_count'})
        )
    else:
        df_counts = (
            df2.groupby([participant_col, format_col], as_index=False)[correct_col]
               .sum().rename(columns={correct_col: 'correct_count'})
        )

    formats = sorted(df_counts[format_col].unique())
    if max_score is None:
        max_score = int(df_counts['correct_count'].max())
    scores = list(range(max_score + 1))

    pivot = (
        df_counts
        .groupby([format_col, 'correct_count']).size()
        .unstack(fill_value=0).reindex(index=formats, fill_value=0)
        .T.reindex(index=scores, fill_value=0)
    )

    # --- 加入 Totals 行列 ---
    pivot_ext = pivot.copy()
    pivot_ext['Total'] = pivot_ext.sum(axis=1)
    total_row = pivot_ext.sum(axis=0)
    pivot_ext.loc['Total'] = total_row

    data = pivot_ext.values
    idx = pivot_ext.index.tolist()
    cols = pivot_ext.columns.tolist()

    # --- 构造两个 mask ---
    mask_main   = np.zeros_like(data, dtype=bool)
    mask_totals = np.zeros_like(data, dtype=bool)
    for i, r in enumerate(idx):
        for j, c in enumerate(cols):
            if r == 'Total' or c == 'Total':
                mask_main[i, j]   = True
            else:
                mask_totals[i, j] = True

    # --- 选择 Totals 区的 cmap ---
    if total_color is None:
        total_cmap = sns.light_palette("gray", as_cmap=True)
    elif isinstance(total_color, str):
        # 如果是一个 colormap 名称
        if total_color in plt.colormaps():
            total_cmap = cm.get_cmap(total_color)
        else:
            # 当作单一颜色，做白到该色渐变
            total_cmap = sns.light_palette(total_color, as_cmap=True)
    elif isinstance(total_color, (list, tuple)):
        total_cmap = sns.color_palette(total_color, as_cmap=True)
    else:
        raise ValueError("total_color 类型不支持。")

    # --- 绘图 ---
    fig, ax = plt.subplots(
        figsize=(1 + len(cols)*0.6, 1 + len(idx)*0.5)
    )
    # 主热力图
    sns.heatmap(
        pivot_ext, mask=mask_main,
        cmap=cmap, annot=annot, fmt=fmt,
        cbar_kws={'label': cbar_label},
        linewidths=0.5, linecolor='gray',
        ax=ax
    )
    # Totals 部分
    sns.heatmap(
        pivot_ext, mask=mask_totals,
        cmap=total_cmap, annot=annot, fmt=fmt,
        cbar=False,
        linewidths=0.5, linecolor='gray',
        ax=ax
    )

    ax.set_xlabel(format_col)
    ax.set_ylabel('Correct Answer Count')
    ax.set_title('Distribution of Correct Answer Counts by Format\n(with Totals)')
    plt.tight_layout()
    plt.show()


def plot_binary_response_vs_metric_heatmap(
    df_post: pd.DataFrame,
    df_result: pd.DataFrame,
    participant_col: str,
    format_col:      str,
    task_col:        str,
    response_col:    str,
    metric_col:      str,
    response_values: list = [0,1],
    cmap:            str  = 'OrRd'
):
    """
    将二元回答 (response_col) 与任意指标 (metric_col) 做透视热力图：
    
    - 行 (index)   : metric_col 的各取值（数值或分类）
    - 列 (columns): response_col 的各取值 (response_values)
    - 格子显示    : 该 (metric, response) 组合的事件/用户数
    
    df_post   要包含 participant_col, format_col, task_col, metric_col
    df_result 要包含 participant_col, format_col, task_col, response_col
    """
    # 1) 合并两张表，保留既有评价又有结果的记录
    df = pd.merge(
        df_post[[participant_col, format_col, task_col, metric_col]],
        df_result[[participant_col, format_col, task_col, response_col]],
        on=[participant_col, format_col, task_col],
        how='inner'
    )

    # 2) 只保留我们关心的二元取值
    df = df[df[response_col].isin(response_values)]

    # 3) 按 (metric, response) 分组计数
    pivot = (
        df
        .groupby([metric_col, response_col])
        .size()
        .unstack(fill_value=0)
    )

    # 4) 确保所有 response_values 都有对应的列
    pivot = pivot.reindex(columns=response_values, fill_value=0)

    # 5) 绘制热力图
    plt.figure(figsize=(6, max(4, pivot.shape[0]*0.5)))
    sns.heatmap(
        pivot,
        annot=True, fmt='d',
        cmap=cmap,
        cbar_kws={'label': f'Count of {response_col}'}
    )
    plt.ylabel(metric_col.replace('_', ' ').title())
    plt.xlabel(response_col.replace('_', ' ').title())
    plt.title(f'Distribution of {response_col.replace("_"," ").title()} vs. {metric_col.replace("_"," ").title()}')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()



def plot_nasatlx_correct(
    df_reading, df_nasa, metrics,
    group_col: str = 'format',
    participant_col: str = 'participantId',
    correct_col: str = 'correct',
    agg_func: str = 'mean',
    show_by_format: bool    = True,
    show_overall: bool      = True,
    show_residual: bool     = True,
    show_overall_trend: bool= True,
    figsize_per: tuple      = (4,4)
):
    """
    修正版：在做 polyfit 和 pearsonr 前，先清洗掉 NaN/Inf，并保证样本量 ≥ 2。
    """
    # —— 1. 本地复制并转换数值型 —— #
    df_r = df_reading.copy()
    df_r[correct_col] = pd.to_numeric(df_r[correct_col], errors='coerce')
    
    # 先把 NASA-TLX 所有指标也转成数值
    df_n = df_nasa.copy()
    for m in metrics:
        df_n[m] = pd.to_numeric(df_n[m], errors='coerce')

    # —— 2. 聚合 NASA-TLX —— #
    if agg_func == 'mean':
        df_n_agg = df_n.groupby(participant_col)[metrics].mean().reset_index()
    else:
        df_n_agg = df_n.groupby(participant_col)[metrics].median().reset_index()

    # —— 3. 合并回答和指标，并初步 drop NaN correct —— #
    df = (
        df_r[[participant_col, correct_col, group_col]]
        .merge(df_n_agg, on=participant_col, how='inner')
        .dropna(subset=[correct_col])
    )

    formats = sorted(df[group_col].dropna().unique())
    cmap    = plt.cm.get_cmap('tab10', len(formats))
    markers = ['o','s','^','D','v','P','X','*','h','8'] * 3

    # —— 4. 针对每个 metric 单独绘图 —— #
    for metric in metrics:
        # 4.1 清洗 (correct, metric) 这两列的 NaN/Inf
        x_all = pd.to_numeric(df[correct_col], errors='coerce').to_numpy()
        y_all = pd.to_numeric(df[metric],       errors='coerce').to_numpy()
        mask  = np.isfinite(x_all) & np.isfinite(y_all)
        x_all = x_all[mask]
        y_all = y_all[mask]

        # 如果样本不足，就跳过
        if len(x_all) < 2:
            print(f"跳过 {metric}: 有效样本 < 2")
            continue

        # 全局拟合
        m_all, b_all = np.polyfit(x_all, y_all, 1)
        y_pred_all   = m_all * x_all + b_all
        residuals    = y_all - y_pred_all
        r_all, p_all = pearsonr(x_all, y_all)

        # 确定子图数量
        flags   = [show_by_format, show_overall, show_residual]
        n_plots = sum(flags)
        if n_plots == 0:
            raise ValueError("至少启用一个 show_* 参数。")

        fig, axes = plt.subplots(
            1, n_plots,
            figsize=(figsize_per[0]*n_plots, figsize_per[1]),
            squeeze=False
        )
        axes = axes[0]
        idx = 0

        # (1) 按 format 的散点 + 回归
        if show_by_format:
            ax = axes[idx]; idx += 1
            for i, fmt in enumerate(formats):
                sub = df[df[group_col] == fmt]
                # 再次清洗子集
                xa = pd.to_numeric(sub[correct_col], errors='coerce').to_numpy()
                ya = pd.to_numeric(sub[metric],       errors='coerce').to_numpy()
                mask_sub = np.isfinite(xa) & np.isfinite(ya)
                xa, ya = xa[mask_sub], ya[mask_sub]
                if len(xa) == 0:
                    continue
                ax.scatter(xa, ya, color=cmap(i), marker=markers[i],
                           label=str(fmt), alpha=0.7)
                if len(xa) >= 2:
                    m, b = np.polyfit(xa, ya, 1)
                    x0 = np.array([xa.min(), xa.max()])
                    ax.plot(x0, m*x0 + b, color=cmap(i), linewidth=1)
            if show_overall_trend:
                x0 = np.array([x_all.min(), x_all.max()])
                ax.plot(x0, m_all*x0 + b_all,
                        color='k', linestyle='--', linewidth=2,
                        label='Overall')
            title = f"{metric}\nBy {group_col}"
            if show_overall_trend:
                title += " + Overall"
            ax.set_title(title)
            ax.set_xlabel('Correct Count')
            ax.set_ylabel(metric)
            ax.legend(title=group_col, bbox_to_anchor=(1.05,1), loc='upper left')

        # (2) overall 散点 + 回归
        if show_overall:
            ax = axes[idx]; idx += 1
            x0 = np.array([x_all.min(), x_all.max()])
            ax.scatter(x_all, y_all, alpha=0.6)
            ax.plot(x0, m_all*x0 + b_all, color='k', linewidth=2)
            ax.set_title(f"{metric}\nOverall Only\nr={r_all:.2f}, p={p_all:.3f}")
            ax.set_xlabel('Correct Count')
            ax.set_ylabel(metric)

        # (3) overall 残差图
        if show_residual:
            ax = axes[idx]; idx += 1
            ax.scatter(x_all, residuals, alpha=0.6)
            ax.axhline(0, color='gray', linewidth=1)
            ax.set_title(f"{metric} Residuals")
            ax.set_xlabel('Correct Count')
            ax.set_ylabel('Residuals')

        plt.tight_layout()
        plt.show()


def plot_time_vs_response_heatmap(
    df_post: pd.DataFrame,
    df_result: pd.DataFrame,
    participant_col: str,
    format_col: str,
    task_col: str,
    time_col: str,
    response_col: str,
    bins: int = 8,
    cmap: str = 'Blues'
):
    """
    1) 合并并清洗
    2) 把 time_col 分成 `bins` 个区间 (quantile bins)
    3) 对 (time_bin, response_col) 做计数透视热力图
    """
    # 合并
    df = pd.merge(
        df_post[[participant_col, format_col, task_col, time_col]],
        df_result[[participant_col, format_col, task_col, response_col]],
        on=[participant_col, format_col, task_col],
        how='inner'
    ).dropna(subset=[time_col, response_col])

    # 分箱（等频）
    df['time_bin'] = pd.qcut(df[time_col], bins, duplicates='drop')

    # 生成透视表
    pivot = (
        df
        .groupby(['time_bin', response_col])
        .size()
        .unstack(fill_value=0)
    )

    # 绘图
    plt.figure(figsize=(6, max(4, pivot.shape[0]*0.5)))
    sns.heatmap(
        pivot, annot=True, fmt='d',
        cmap=cmap,
        cbar_kws={'label': f'Count of {response_col}'}
    )
    plt.ylabel(f"{time_col} Bins")
    plt.xlabel(response_col)
    plt.title(f"{response_col} vs. Binned {time_col}")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()



def plot_duration_vs_correct_with_stats(
    df_post, df_result,
    participant_col='participantId',
    format_col='format',
    task_col='task',
    time_col='duration_sec',
    response_col='correct',
    by_format=True,
    figsize=(16, 4),
    test='mannwhitney'
):
    """
    按 format 分面画箱线，并在每个子图上标注：
      1) 两组间的 p-value（Mann-Whitney U 或 t-test）
      2) 对应的点二列相关系数 r
    
    参数:
      test: 'mannwhitney' 或 'ttest'
    """
    # 合并
    df = (pd.merge(
            df_post[[participant_col, format_col, task_col, time_col]],
            df_result[[participant_col, format_col, task_col, response_col]],
            on=[participant_col, format_col, task_col], how='inner')
          .dropna(subset=[time_col, response_col]))
    
    formats = sorted(df[format_col].unique())
    n = len(formats)
    fig, axes = plt.subplots(1, n, figsize=(figsize[0], figsize[1]), sharey=False)
    
    for ax, fmt in zip(axes, formats):
        sub = df[df[format_col]==fmt]
        # 箱线
        sns.boxplot(data=sub, x=response_col, y=time_col, ax=ax)
        ax.set_title(f"{fmt}", fontsize=12)
        ax.set_xlabel('')
        ax.set_ylabel(time_col if fmt==formats[0] else '')
        
        # 取两组数据
        x0 = sub[sub[response_col]==0][time_col]
        x1 = sub[sub[response_col]==1][time_col]
        
        # 1) 计算 p 值
        if test=='ttest':
            stat, p = st.ttest_ind(x0, x1, nan_policy='omit')
        else:
            stat, p = st.mannwhitneyu(x0, x1, alternative='two-sided')
        
        # 2) 计算点二列相关系数
        r, _ = st.pointbiserialr(sub[response_col], sub[time_col])
        
        # 叠加文本
        ax.text(
            0.5, 0.95,
            f"p={p:.3f}\nr={r:.2f}",
            ha='center', va='top', 
            transform=ax.transAxes,
            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='gray', alpha=0.5)
        )
    
    fig.suptitle(f"{time_col} by {response_col} and {format_col}", y=1.02)
    plt.tight_layout()
    plt.show()


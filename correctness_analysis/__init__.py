import pandas as pd
import re
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
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
    画一个“任务 × 格式” 的正确率热力图，并可通过 task_list / format_list 有选择地只看子集。

    参数:
      df            : 原始数据 DataFrame
      task_col      : 任务名称所在列
      format_col    : 格式名称所在列
      accuracy_col  : 0–1 之间的正确率列
      task_list     : 若非 None，只可视化此列表中的任务
      format_list   : 若非 None，只可视化此列表中的格式
      cmap          : seaborn/matplotlib colormap 名称
      annot         : 是否在每个格子里写上数值
      fmt           : annot 的数字格式
      cbar_label    : 右侧 colorbar 的标题
    """
    df2 = df.copy()

    # —— 可选过滤 —— 
    if task_list is not None:
        df2 = df2[df2[task_col].isin(task_list)]
    if format_list is not None:
        df2 = df2[df2[format_col].isin(format_list)]
    # 如果过滤后为空，提示一下
    if df2.empty:
        raise ValueError("过滤后没有数据: 请检查 task_list / format_list 是否正确。")

    # —— 计算 pivot 矩阵 —— 
    pivot = (
        df2
        .groupby([task_col, format_col])[accuracy_col]
        .mean()
        .unstack(fill_value=0)   # 对于缺失的组合填 0
    )

    # —— 绘图尺寸 —— 
    n_tasks  = pivot.shape[0]
    n_formats= pivot.shape[1]
    figsize = (max(6, n_formats*1.2), max(4, n_tasks*0.6))

    # —— 绘制 heatmap —— 
    plt.figure(figsize=figsize)
    ax = sns.heatmap(
        pivot,
        cmap=cmap,
        annot=annot,
        fmt=fmt,
        cbar_kws={'label': cbar_label},
        linewidths=.5,
        linecolor='gray'
    )

    ax.set_xlabel(format_col)
    ax.set_ylabel(task_col)
    ax.set_title("Accuracy Heatmap: Task vs Format")
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
    cbar_label: str = 'Number of Participants'
):
    """
    画“格式 × 答对题数”的分布热力图：
      • 确定每人每格式的总分：  
        – 如果有 task=='task_all'，直接用对应 correct 作为 correct_count  
        – 否则把各小题(correct=0/1)按 (participant,format) 分组求和  
      • 横轴：各序列化格式  
      • 纵轴：答对题数（0..max_score）  
      • 格子值：恰好答对该题数的人数  
    """
    df2 = df.copy()

    # 1) 计算每个人每格式的总分 correct_count
    if 'task_all' in df2[task_col].unique():
        df_counts = (
            df2[df2[task_col] == 'task_all']
            [[participant_col, format_col, correct_col]]
            .rename(columns={correct_col: 'correct_count'})
        )
    else:
        df_counts = (
            df2
            .groupby([participant_col, format_col], as_index=False)[correct_col]
            .sum()
            .rename(columns={correct_col: 'correct_count'})
        )

    # 2) 确定可视化的格式列表和分数范围
    formats = sorted(df_counts[format_col].unique())
    if max_score is None:
        max_score = int(df_counts['correct_count'].max())
    scores = list(range(0, max_score + 1))

    # 3) 统计每个 (format, score) 的人数
    pivot = (
        df_counts
        .groupby([format_col, 'correct_count'])
        .size()
        .unstack(fill_value=0)      # 行=index=format，列=correct_count
        .reindex(index=formats, fill_value=0)
        .T                          # 转置：行→correct_count，列→format
        .reindex(index=scores, fill_value=0)
    )

    # 4) 绘制热力图
    plt.figure(figsize=(1 + len(formats)*0.6, 1 + len(scores)*0.5))
    ax = sns.heatmap(
        pivot,
        cmap=cmap,
        annot=annot,
        fmt=fmt,
        cbar_kws={'label': cbar_label},
        linewidths=0.5,
        linecolor='gray'
    )
    ax.set_xlabel(format_col)
    ax.set_ylabel('Correct Answer Count')
    ax.set_title('Distribution of Correct Answer Counts by Format')
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
    Flexible plotting of Correct vs metrics:
      - show_by_format:  by-group scatter + per-group regressions (＋optional overall line)
      - show_overall:    overall scatter + regression
      - show_residual:   residuals vs correct

    参数:
      df_reading, df_nasa: DataFrames
      metrics: list of metric names
      group_col: 分组列名
      ...
      show_overall_trend: 当 show_by_format=True 时，是否绘制 overall 虚线
      figsize_per: 每个子图的大小 (w,h)
    """
    df_reading['correct'] = pd.to_numeric(df_reading['correct'], errors='coerce')
    for col in ['mental-demand','physical-demand','temporal-demand','effort','performance','frustration']:
        df_nasa[col] = pd.to_numeric(df_nasa[col], errors='coerce')
    # 1. 聚合 NASA 数据
    if agg_func=='mean':
        df_nasa_agg = df_nasa.groupby(participant_col)[metrics].mean().reset_index()
    else:
        df_nasa_agg = df_nasa.groupby(participant_col)[metrics].median().reset_index()

    # 2. 合并
    df = (df_reading[[participant_col, correct_col, group_col]]
          .merge(df_nasa_agg, on=participant_col, how='inner')
          .dropna(subset=[correct_col]))

    formats = sorted(df[group_col].unique())
    cmap    = plt.cm.get_cmap('tab10', len(formats))
    markers = ['o','s','^','D','v','P','X','*','h','8'] * 3

    # 3. 针对每个 metric
    for metric in metrics:
        # 计算 overall 回归参数和残差数据
        x_all = df[correct_col]; y_all = df[metric]
        m_all, b_all = np.polyfit(x_all, y_all, 1)
        y_pred_all = m_all * x_all + b_all
        residuals  = y_all - y_pred_all
        r_all, p_all = pearsonr(x_all, y_all)

        # 确定要画多少图
        flags = [show_by_format, show_overall, show_residual]
        n_plots = sum(flags)
        if n_plots == 0:
            raise ValueError("至少启用一个 show_* 参数。")
        fig, axes = plt.subplots(1, n_plots,
                                 figsize=(figsize_per[0]*n_plots, figsize_per[1]),
                                 squeeze=False)
        axes = axes[0]

        idx = 0
        # (1) 按 format
        if show_by_format:
            ax = axes[idx]; idx += 1
            for i, fmt in enumerate(formats):
                sub = df[df[group_col]==fmt]
                x, y = sub[correct_col], sub[metric]
                ax.scatter(x, y, color=cmap(i), marker=markers[i],
                           label=str(fmt), alpha=0.7)
                if len(sub)>=2:
                    m, b = np.polyfit(x, y, 1)
                    x0 = np.array([x.min(), x.max()])
                    ax.plot(x0, m*x0+b, color=cmap(i), linewidth=1)
            if show_overall_trend:
                x0 = np.array([x_all.min(), x_all.max()])
                ax.plot(x0, m_all*x0+b_all,
                        color='k', linestyle='--', linewidth=2,
                        label='Overall')
            ax.set_title(f"{metric}\nBy {group_col}")
            if show_overall_trend:
                ax.set_title(f"{metric}\nBy {group_col} + Overall")
            ax.set_xlabel('Correct Count')
            ax.set_ylabel(metric)
            ax.legend(title=group_col, bbox_to_anchor=(1.05,1), loc='upper left')

        # (2) overall only
        if show_overall:
            ax = axes[idx]; idx += 1
            x0 = np.array([x_all.min(), x_all.max()])
            ax.scatter(x_all, y_all, alpha=0.6)
            ax.plot(x0, m_all*x0+b_all, color='k', linewidth=2)
            ax.set_title(f"{metric}\nOverall Only\nr={r_all:.2f}, p={p_all:.3f}")
            ax.set_xlabel('Correct Count')
            ax.set_ylabel(metric)

        # (3) residuals
        if show_residual:
            ax = axes[idx]; idx += 1
            ax.scatter(x_all, residuals, alpha=0.6)
            ax.axhline(0, color='gray', linewidth=1)
            ax.set_title(f"{metric} Residuals")
            ax.set_xlabel('Correct Count')
            ax.set_ylabel('Residuals')

        plt.tight_layout()
        plt.show()

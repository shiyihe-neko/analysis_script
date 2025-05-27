import pandas as pd
import re
import os
import json
import numpy as np
from typing import Tuple, List, Optional
import matplotlib.pyplot as plt


def time_analysis(all_data, metric='total_duration_min'):
    """
    Parameters:
      all_data (dict): The dictionary returned by load_quiz_data.

    Returns:
      df_task_level (pandas.DataFrame):
        A DataFrame with one row per task, including columns for start time, end time, duration, and format.
      df_participant_level (pandas.DataFrame):
        A DataFrame with one row per participant, including columns for total duration and format.
    """
    task_rows = []
    part_rows = []

    for fn, quiz in all_data.items():
        answers = quiz.get('answers', {})
        # look for participantId
        pid = fn
        for info in answers.values():
            if isinstance(info, dict):
                ans = info.get('answer', {})
                if isinstance(ans, dict) and 'prolificId' in ans:
                    pid = ans['prolificId']
                    break

        # extract  format
        current_format = None
        total_sec = 0
        temp = []
        for name, info in answers.items():
            if not isinstance(info, dict):
                continue
            st = info.get('startTime')
            ed = info.get('endTime')
            if st is not None and ed is not None:
                dur = (ed - st)/1000.0
            else:
                dur = None

            # tutorial-<fmt>-part1
            if current_format is None:
                m = re.match(r'tutorial-(\w+)-part1', name)
                if m:
                    current_format = m.group(1).lower()

            temp.append({
                'participantId': pid,
                'task': name,
                'startTime': st,
                'endTime': ed,
                'duration_sec': dur,
                'duration_min': dur/60 if dur is not None else None
            })
            if dur:
                total_sec += dur

        # add format 
        fmt = current_format or 'unknown'
        for row in temp:
            row['format'] = fmt
            task_rows.append(row)

        part_rows.append({
            'participantId': pid,
            'format': fmt,
            'duration_sec': round(total_sec,3),
            'duration_min': round(total_sec/60,2)
        })

    df_task = pd.DataFrame(task_rows)
    df_part = pd.DataFrame(part_rows)
    def clean(name, fmt):
        if fmt and fmt!='unknown':
            return name.replace(f"-{fmt}", "")
        return name

    df_task['task'] = df_task.apply(lambda r: clean(r['task'], r['format']), axis=1)
    format_stat = df_part['format'].value_counts()
    total_participant = len(df_part)
    print(f"Total number of valid participants: {total_participant}")

    return df_task, df_part, format_stat



def normalize_typing_time(df_nl: pd.DataFrame, df_tab: pd.DataFrame,
                          keep_baseline: bool = True) -> pd.DataFrame:
    """
        normalized_time = duration_sec / char_per_sec

    参数：
    - df_nl: baseline 
    - df_tab: writing tabular or config
    - keep_baseline: 是否返回 baseline 数据一并合并

    返回：
    - if keep_baseline=True, return baseline + writing task merged DataFrame
    - else: task only normalized_time 
    """

    # 1. baseline typing speed
    df_nl2 = df_nl.copy()
    df_nl2['char_count']    = df_nl2['code'].str.len()
    df_nl2['char_per_sec']  = df_nl2['char_count'] / df_nl2['duration_sec']

    # 2. 归一化任务：merge baseline speed
    df_tab2 = df_tab.copy()
    df_tab2['char_count'] = df_tab2['code'].str.len()

    df_tab2 = df_tab2.merge(
        df_nl2[['participantId', 'char_per_sec']],
        on='participantId',
        how='left'
    )

    # 3. normalized time = duration_sec / char_per_sec
    df_tab2['normalized_time'] = df_tab2['duration_sec'] / df_tab2['char_per_sec']

    # 4. return
    if keep_baseline:
        return pd.concat([df_nl2, df_tab2], ignore_index=True, sort=False)
    else:
        return df_tab2




def compute_task_time_ratio(df_total: pd.DataFrame,
                            df_task: pd.DataFrame,
                            duration_total_col: str = 'duration_sec',
                            duration_task_col: str = 'duration_sec') -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    计算每个参与者 task 时间占总时间的比例

    参数：
    - df_total: 总时间表（每个 participant 的总时长）
    - df_task: 某类任务时间表（例如 writing / modifying）
    - duration_total_col: df_total 中的总时间字段名（默认 'duration_sec'）
    - duration_task_col: df_task 中的 task 时间字段名（默认 'duration_sec'）

    返回：
    - df_ratio: 包含 participantId, task_time, total_time, format, ratio
    - df_format_avg: 每种 format 的平均 ratio
    """
    
    # 对 df_task 汇总每人每个 format 的 task 总时间
    df_task_sum = (
        df_task
        .groupby(['participantId', 'format'])[duration_task_col]
        .sum()
        .reset_index()
        .rename(columns={duration_task_col: 'task_time'})
    )

    # 合并总时间
    df_merged = df_task_sum.merge(
        df_total[['participantId', duration_total_col]],
        on='participantId',
        how='left'
    ).rename(columns={duration_total_col: 'total_time'})

    # 计算比例
    df_merged['ratio'] = df_merged['task_time'] / df_merged['total_time']

    df_merged['task'] = 'writing_time/total_time'

    # 每种格式的平均比例
    df_format_avg = (
        df_merged
        .groupby('format')['ratio']
        .mean()
        .reset_index()
        .rename(columns={'ratio': 'avg_ratio'})
    )

    return df_merged[['participantId', 'format', 'task_time', 'total_time', 'ratio','task']], df_format_avg

def get_task_group(task: str) -> str:
    if task.endswith('_post-task-question'):
        return 'post-task-question'
    m = re.match(r'^(.*?)(?:-\d+|-part\d+)$', task)
    if m:
        return m.group(1)
    return task

def compute_avg_ratio(
    df: pd.DataFrame,
    tasks: Optional[List[str]] = None
) -> (pd.DataFrame, pd.DataFrame):
    df = df.copy()
    df['task_group'] = df['task'].apply(get_task_group)
    total = (
        df.groupby(['participantId','format'], as_index=False)
          .agg(total_sec=('duration_sec','sum'))
    )
    grp = (
        df.groupby(['participantId','format','task_group'], as_index=False)
          .agg(group_sec=('duration_sec','sum'))
    )
    dfp = (
        grp.merge(total, on=['participantId','format'])
           .assign(ratio=lambda d: d['group_sec']/d['total_sec'])
    )
    if tasks is None:
        df_avg = (
            dfp.groupby(['format','task_group'], as_index=False)
               .agg(avg_ratio=('ratio','mean'))
               .sort_values('avg_ratio', ascending=False)
               .reset_index(drop=True)
        )
        wide_avg = df_avg.pivot(index='format', columns='task_group', values='avg_ratio').fillna(0)
    else:
        df_sel = dfp[dfp['task_group'].isin(tasks)]
        summed = (
            df_sel.groupby(['participantId','format'], as_index=False)
                  .agg(ratio_sum=('ratio','sum'))
        )
        df_avg = (
            summed.groupby('format', as_index=False)
                  .agg(avg_ratio=('ratio_sum','mean'))
                  .sort_values('avg_ratio', ascending=False)
                  .reset_index(drop=True)
        )
        wide_avg = df_avg.set_index('format')[['avg_ratio']]
    return df_avg, wide_avg

def plot_stacked_ratio(
    all_wide,
    figsize=(10, 6),
    sort_formats_by=None,
    stack_order_by=None,
    cmap_name='tab20'
):
    """
    绘制堆叠柱状图，并可：
      1. sort_formats_by： 按某个 task_group 的占比给 x 轴的 formats 排序
      2. stack_order_by：  按某个 format（或 'average'）给 stack segments 排序

    现在固定 format 显示顺序：['json','json5','jsonc','xml','yaml','toml','hjson']
    """
    # 定义固定顺序
    order = ['json', 'jsonc', 'json5', 'hjson', 'xml', 'yaml', 'toml']

    # 复制数据
    df = all_wide.copy()
    # 仅保留预设顺序中的 formats
    df = df.loc[df.index.intersection(order)]

    # —— 1) 按 formats 排序 —— #
    if sort_formats_by and sort_formats_by in df.columns:
        # 按指定列排序
        df = df.sort_values(by=sort_formats_by, ascending=False)
        formats = df.index.tolist()
    else:
        # 使用预定顺序
        # 将索引设为分类索引以按 order 排列
        df.index = pd.CategoricalIndex(df.index, categories=order, ordered=True)
        df = df.sort_index()
        formats = order

    # —— 2) 决定 stack 的顺序 —— #
    if stack_order_by == 'average':
        base = df.mean(axis=0)
    elif stack_order_by in df.index:
        base = df.loc[stack_order_by]
    else:
        base = None

    if base is not None:
        task_groups = base.sort_values().index.tolist()
    else:
        task_groups = df.columns.tolist()

    # 统一调色
    cmap = plt.get_cmap(cmap_name)
    colors = cmap(np.linspace(0, 1, len(task_groups)))

    # —— 3) 绘图 —— #
    bottom = np.zeros(len(formats))
    fig, ax = plt.subplots(figsize=figsize)
    for tg, color in zip(task_groups, colors):
        vals = df[tg].reindex(formats).values
        ax.bar(formats, vals, bottom=bottom, label=tg, color=color)
        bottom += vals

    # —— 4) 美化 —— #
    ax.set_xlabel('Format')
    ax.set_ylabel('Average Time Ratio')
    ax.set_title('Stacked Bar Chart of Task Time Ratios by Format')
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1))
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    return fig, ax

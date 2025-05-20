import os
import json
import re
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from typing import Tuple, Dict
from typing import Union

def load_all_data(folder_path, ignore_completed=False):
    """
    Iterate over all .json files in folder_path:
      - By default, include only those with quiz['completed'] == True;
      - Rename answer keys that have numeric suffixes (_1, _2, …)
      - Apply renaming logic for post-task-question and post-task-survey keys

    Returns a dict mapping each filename (without extension) to its processed quiz data dict.
    """
    def extract_suffix(key):
        m = re.search(r'_(\d+)$', key)
        return int(m.group(1)) if m else 0

    def remove_suffix(key):
        return re.sub(r'_(\d+)$', '', key)

    all_data = {}
    for fn in os.listdir(folder_path):
        if not fn.lower().endswith('.json'):
            continue
        path = os.path.join(folder_path, fn)
        try:
            with open(path, encoding='utf-8') as f:
                quiz = json.load(f)
        except json.JSONDecodeError:
            continue

        if not ignore_completed and not quiz.get('completed', False):
            continue

        key_name = os.path.splitext(fn)[0]
        all_data[key_name] = quiz

        answers = quiz.get('answers', {})
        if not isinstance(answers, dict):
            continue

        sorted_keys = sorted(answers.keys(), key=extract_suffix)
        new_answers = {}
        last_task = None
        for i, old in enumerate(sorted_keys):
            base = remove_suffix(old)

            if base == 'post-task-question':
                new_key = f"{last_task}_post-task-question" if last_task else base
            elif base.startswith('post-task-survey'):
                if i > 0:
                    prev = sorted_keys[i-1]
                    prev_base = remove_suffix(prev)
                    suffix = prev_base[prev_base.rfind('-'):] if '-' in prev_base else ''
                    new_key = base + suffix
                else:
                    new_key = base
                last_task = None
            else:
                new_key = base
                last_task = base

            new_answers[new_key] = answers[old]

        quiz['answers'] = new_answers

    return all_data



def _get_participant_id(answers: dict) -> str:
    for content in answers.values():
        if not isinstance(content, dict):
            continue
        ans = content.get('answer', {}) or {}
        if isinstance(ans, dict) and 'prolificId' in ans:
            return ans['prolificId']
    return None


def extract_writing_nl_tasks(all_data: dict) -> pd.DataFrame:
    """
    extract writing-task-NL task

    return: participant_id, component, format='NL',
              code, start_time, end_time, duration_sec, help_count
    """
    rows = []
    for session in all_data.values():
        answers = session.get('answers', {})
        pid = _get_participant_id(answers)
        content = answers.get('writing-task-NL')
        if isinstance(content, dict):
            st = content.get('startTime')
            ed = content.get('endTime')
            dur = (ed - st) / 1000.0 if st is not None and ed is not None else None
            ans = content.get('answer', {}) or {}
            code = ans.get('code')
            help_count = content.get('helpButtonClickedCount')
            rows.append({
                'participantId': pid,
                'task':      'writing-task-NL',
                'format':         'NL',
                'code':           code,
                'start_time':     st,
                'end_time':       ed,
                'duration_sec':   dur,
                'help_count':     help_count
            })
    return pd.DataFrame(rows)


def extract_reading_tasks(all_data: dict) -> pd.DataFrame:
    """
    extract reading-task-<type>-<format>-<number> task

    return
      participant_id, task, format,
      response, start_time, end_time, duration_sec, help_count
    """
    rows = []
    for _, session in all_data.items():
        answers = session.get('answers', {})
        pid = _get_participant_id(answers)

        for content in answers.values():
            if not isinstance(content, dict):
                continue

            name = content.get('componentName', '')

            # 检查格式 reading-task-<type>-<format>-<number>
            parts = name.split('-')
            if name.startswith('reading-task-') and len(parts) == 5:
                task_type = parts[2]    # tabular / config / ...
                fmt = parts[3]          # JSON / YAML / ...
                num = parts[4]          # 问题编号

                # 构造 q_key
                q_key = f"reading-task-{task_type}-{fmt}_q{num}"
                ans = content.get('answer', {}) or {}
                resp = ans.get(q_key)

                st = content.get('startTime')
                ed = content.get('endTime')
                dur = (ed - st) / 1000.0 if st is not None and ed is not None else None
                help_count = content.get('helpButtonClickedCount')

                rows.append({
                    'participantId': pid,
                    'task':          name,
                    'format':        fmt,
                    'response':      resp,
                    'start_time':    st,
                    'end_time':      ed,
                    'duration_sec':  dur,
                    'help_count':    help_count
                })

        df= pd.DataFrame(rows)        
    
        def _strip(row):
            task = row['task']
            parts = task.split('-')
            if len(parts) == 5:
                # 移除第4段（格式）
                parts.pop(3)
                return '-'.join(parts)
            return task  # fallback
        
        df['task'] = df.apply(_strip, axis=1)

    return df


def extract_writing_tasks(all_data: dict) -> pd.DataFrame:
    """
    extract writing-task-<type>-<format> task
    e.g., writing-task-tabular-JSON, writing-task-config-YAML

    return: participant_id, component, format,
              code, start_time, end_time, duration_sec, help_count
    """
    rows = []
    for session in all_data.values():
        answers = session.get('answers', {})
        pid = _get_participant_id(answers)

        for content in answers.values():
            if not isinstance(content, dict):
                continue

            name = content.get('componentName', '')

            # 匹配 writing-task-<type>-<format>（非NL）
            if name.startswith('writing-task-') and name.count('-') == 3:
                # 获取格式（如 JSON、YAML）
                fmt = name.split('-')[-1]

                st = content.get('startTime')
                ed = content.get('endTime')
                dur = (ed - st) / 1000.0 if st is not None and ed is not None else None

                ans = content.get('answer', {}) or {}
                code = ans.get('code')
                help_count = content.get('helpButtonClickedCount')

                rows.append({
                    'participantId': pid,
                    'task':          name,
                    'format':        fmt,
                    'code':          code,
                    'start_time':    st,
                    'end_time':      ed,
                    'duration_sec':  dur,
                    'help_count':    help_count
                })
        df= pd.DataFrame(rows)        
    
        def _strip(row):
            task = row['task']
            fmt  = row['format']
            return re.sub(f'-{re.escape(fmt)}$', '', task)
        df['task'] = df.apply(_strip, axis=1)

    return df



def extract_modifying_tasks(all_data: dict) -> pd.DataFrame:
    """
    extract modifying-task-<type>-<format>-<number> tasks，
    like: modifying-task-tabular-JSON-1, modifying-task-config-YAML-2 

    return: participantId, task, format, code, start_time, end_time, duration_sec, help_count
    """
    rows = []
    for session in all_data.values():
        answers = session.get('answers', {})
        pid = _get_participant_id(answers)

        for content in answers.values():
            if not isinstance(content, dict):
                continue

            name = content.get('componentName', '')

            # 检查是否是符合 modifying-task-<type>-<format>-<number> 格式
            parts = name.split('-')
            if name.startswith('modifying-task-') and len(parts) == 5:
                fmt = parts[-2]  # 倒数第二个部分是 format（如 JSON、YAML）

                st = content.get('startTime')
                ed = content.get('endTime')
                dur = (ed - st) / 1000.0 if st is not None and ed is not None else None

                ans = content.get('answer', {}) or {}
                code = ans.get('code')
                help_count = content.get('helpButtonClickedCount')

                rows.append({
                    'participantId': pid,
                    'task':          name,
                    'format':        fmt,
                    'code':          code,
                    'start_time':    st,
                    'end_time':      ed,
                    'duration_sec':  dur,
                    'help_count':    help_count
                })

        df= pd.DataFrame(rows)        
    
        def _strip(row):
            task = row['task']
            parts = task.split('-')
            if len(parts) == 5:
                # 移除第4段（格式）
                parts.pop(3)
                return '-'.join(parts)
            return task  # fallback
        
        df['task'] = df.apply(_strip, axis=1)

    return df

def extract_post_task_questions(all_data: dict) -> pd.DataFrame:
    """
    从 all_data 中提取 *_post-task-question，输出：
      ['participantId','format','task','startTime','endTime',
       'duration_sec','difficulty','confidence']
    最后一步会把 task 名称中的 "-<format>" （可跟 "-数字"）都去除。
    """
    rows = []
    for file_name, quiz_data in all_data.items():
        answers = quiz_data.get('answers', {})

        # 1) participantId
        pid = file_name
        for info in answers.values():
            if isinstance(info, dict):
                a = info.get('answer', {}) or {}
                if 'prolificId' in a:
                    pid = a['prolificId']
                    break

        # 2) format，从 tutorial-<fmt>-part1 里提取
        fmt = "unknown"
        for k in answers:
            m = re.match(r"tutorial-(\w+)-part1", k)
            if m:
                fmt = m.group(1).lower()
                break

        # 3) 扫描所有 post-task-question 项
        for key, content in answers.items():
            if not key.endswith('_post-task-question'):
                continue
            if not isinstance(content, dict):
                continue

            task_name = key[:-len('_post-task-question')]
            ans = content.get('answer', {}) or {}
            st, ed = content.get('startTime'), content.get('endTime')
            dur = (ed - st)/1000.0 if (st and ed) else None
            diff = ans.get('difficulty')
            conf = ans.get('confidence')

            rows.append({
                'participantId': pid,
                'format':        fmt,
                'task':          task_name,
                'startTime':     st,
                'endTime':       ed,
                'duration_sec':  dur,
                'difficulty':    diff,
                'confidence':    conf
            })

    df = pd.DataFrame(rows)

    # 4) 数值列强制转换
    for c in ['duration_sec','difficulty','confidence']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # 5) 动态生成要清除的格式列表
    formats = list(df['format'].dropna().unique())
    # 为了让长的格式名（如 jsonc、json5、hjson）优先匹配，按长度降序
    formats.sort(key=len, reverse=True)

    # 构造正则：-(?:fmt1|fmt2|...)(?=(?:-\d+$)|$)
    fmt_pat = '|'.join(re.escape(f) for f in formats)
    regex   = fr'-(?:{fmt_pat})(?=(?:-\d+$)|$)'

    # 6) 清洗 task 列
    df['task'] = df['task'].str.replace(regex, '', regex=True)

    return df


def melt_tlx(
    df: pd.DataFrame,
    id_vars: list = None,
    tlx_vars: list = None,
    var_name: str = 'task',
    value_name: str = 'score'
) -> pd.DataFrame:
    """
    将 NASA-TLX 六个维度从宽表转成长表：
      mental-demand, physical-demand, temporal-demand,
      performance, effort, frustration
    会被融入一列 `var_name`，对应的数值列为 `value_name`。

    参数:
      df:        原始 DataFrame，需含上述维度列
      id_vars:   保留作为标识符的列列表，默认 ['participantId','format','startTime','endTime','duration_sec']
      tlx_vars:  待转换的列列表，默认上述六个
      var_name:  新列名，存放原列名
      value_name:新列名，存放原值

    返回:
      长表格式的 DataFrame，列为 id_vars + [var_name, value_name]
    """
    # 默认保留列
    if id_vars is None:
        id_vars = ['participantId', 'format', 'startTime', 'endTime', 'duration_sec']
    # 默认待融入列
    if tlx_vars is None:
        tlx_vars = [
            'mental-demand',
            'physical-demand',
            'temporal-demand',
            'performance',
            'effort',
            'frustration',
        ]
    # 使用 pandas.melt
    df_long = df.melt(
        id_vars=id_vars,
        value_vars=tlx_vars,
        var_name=var_name,
        value_name=value_name
    )
    df_long['score'] = pd.to_numeric(df_long['score'], errors='coerce')
    return df_long

def analyze_nasa_and_post_surveys(all_data):
    """
    Analyze NASA-TLX and post-task surveys:
      - Extract participantId and format from tutorial keys
      - Build two DataFrames: df_nasa (NASA-TLX) and df_post (post-task survey)
    Returns:
      df_nasa: columns=['participantId','format','startTime','endTime','duration_sec',
                        'mental-demand','physical-demand','temporal-demand',
                        'performance','effort','frustration']
      df_post: columns=[…post-task survey fields…]
    """
    nasa_rows = []
    post_survey_rows = []

    for file_name, quiz_data in all_data.items():
        answers = quiz_data.get('answers', {})

        # participantId
        pid = file_name
        for info in answers.values():
            if isinstance(info, dict):
                ans = info.get('answer', {})
                if isinstance(ans, dict) and 'prolificId' in ans:
                    pid = ans['prolificId']
                    break

        # format
        fmt = "unknown"
        for k in answers:
            m = re.match(r"tutorial-(\w+)-part1", k)
            if m:
                fmt = m.group(1).lower()
                break

        # NASA-TLX
        key = '$nasa-tlx.co.nasa-tlx'
        if key in answers:
            info = answers[key]
            ans = info.get('answer', {})
            st, ed = info.get('startTime'), info.get('endTime')
            dur = (ed-st)/1000.0 if st and ed else None
            row = {
                'participantId': pid,
                'format': fmt,
                'startTime': st,
                'endTime': ed,
                'duration_sec': dur
            }
            for dim in ['mental-demand','physical-demand','temporal-demand',
                        'performance','effort','frustration']:
                row[dim] = ans.get(dim)
            nasa_rows.append(row)

    df_nasa = pd.DataFrame(nasa_rows)
    return df_nasa


def merge_dfs(df1: pd.DataFrame, df2: pd.DataFrame,
              on_cols: list, how: str = 'inner') -> pd.DataFrame:

    merged = pd.merge(df1, df2, on=on_cols, how=how)
    return merged



def aggregate_tasks_with_format(
    df: pd.DataFrame,
    participant_col: str = 'participant_id',
    task_col: str = 'task',
    format_col: str = 'format',
    metrics: list = ['duration_sec', 'help_count']
) -> pd.DataFrame:
    df = df.copy()

    # 1) 筛选末尾带 -数字 的子任务
    mask = df[task_col].str.contains(r"-\d+$", regex=True)
    sub = df[mask].copy()

    # 2) 计算 prefix = 去掉尾部 '-number'
    sub_prefix = sub.copy()
    sub_prefix[task_col] = sub_prefix[task_col].str.replace(r"-\d+$", "", regex=True)

    # 3) 按 participant + format + prefix 聚合 metrics
    agg = (
        sub_prefix
        .groupby([participant_col, format_col, task_col], as_index=False)[metrics]
        .sum()
    )

    # 4) 合并原始行和聚合行
    result = pd.concat([df, agg], ignore_index=True, sort=False)
    return result



def aggregate_quiz_parts(
    df: pd.DataFrame,
    participant_col: str = 'participantId',
    format_col: str = 'format',
    task_col: str = 'quiz_key',
    sum_cols: list = None,
    mean_cols: list = None
) -> pd.DataFrame:
    df2 = df.copy()
    sum_cols = sum_cols or []
    mean_cols = mean_cols or []

    # 1) 计算基准任务名：去掉末尾 '-part1' 或 '-part2'
    base = df2[task_col].str.replace(r'-part[12]$', '', regex=True)

    # 2) 构建聚合映射
    agg_map = {c: 'sum' for c in sum_cols}
    agg_map.update({c: 'mean' for c in mean_cols})

    # 3) 按 participant, format, base 进行聚合
    agg = (
        df2.assign(**{'_base': base})
           .groupby([participant_col, format_col, '_base'], as_index=False)
           .agg(agg_map)
           .rename(columns={'_base': task_col})
    )

    # 4) 合并 原始行 + 聚合行，不改变原始数据
    combined = pd.concat([df2, agg], ignore_index=True, sort=False)
    return combined


def summarize_quiz_parts(
    df: pd.DataFrame,
    participant_col: str = 'participantId',
    format_col:    str = 'format',
    task_col:      str = 'quiz_key',
    sum_cols:      list = None,
    mean_cols:     list = None
) -> pd.DataFrame:
    df2 = df.copy()
    sum_cols  = sum_cols or []
    mean_cols = mean_cols or []

    # 1) 先生成“基准任务名”：去掉尾部 '-part1' 或 '-part2'
    base = df2[task_col].str.replace(r'-part[12]$', '', regex=True)

    # 2) 构建聚合映射
    agg_map = {c: 'sum'  for c in sum_cols}
    agg_map.update({c: 'mean' for c in mean_cols})

    # 3) 分组并聚合，只保留汇总行
    agg = (
        df2
        .assign(_base=base)
        .groupby([participant_col, format_col, '_base'], as_index=False)
        .agg(agg_map)
        .rename(columns={'_base': task_col})
    )

    # 4) 返回聚合结果
    return agg


def evaluate_quiz_answers_from_tutorial(all_data):
    """
    Iterate through all_data to evaluate tutorial quiz answers:
      - Extract participantId from prolificId or filename
      - Extract format from tutorial-<format>-part1 key
      - For each tutorial part (part1/part2), compare user answers against correct answers
      - Count wrong attempts and distributions
      - Return a DataFrame of quiz results per participant, quiz key, and format
    """
    quiz_results = []

    for file_name, quiz_data in all_data.items():
        answers = quiz_data.get('answers', {})

        # 1. 提取 participantId
        participant_id = None
        for task_info in answers.values():
            if isinstance(task_info, dict):
                answer_block = task_info.get('answer', {})
                if isinstance(answer_block, dict) and 'prolificId' in answer_block:
                    participant_id = answer_block['prolificId']
                    break
        if participant_id is None:
            participant_id = file_name

        # 2. 提取 format
        format_name = "unknown"
        for k in answers.keys():
            m = re.match(r"tutorial-(\w+)-part1", k)
            if m:
                format_name = m.group(1).lower()
                break

        # 3. 遍历每个 quiz 任务
        for task_key, task_info in answers.items():
            if not isinstance(task_info, dict):
                continue
            if not re.match(r"tutorial-\w+-part[12]$", task_key):
                continue

            # 3.1 拿到正确答案
            correct_ans_list = task_info.get("correctAnswer", [])
            if (not correct_ans_list
                    or not isinstance(correct_ans_list[0], dict)):
                continue
            quiz_id = correct_ans_list[0].get("id")
            correct_answer = correct_ans_list[0].get("answer", [])
            correct_set = set(correct_answer)

            # 3.2 拿到用户最终答案
            answer_block = task_info.get("answer", {})
            user_final_ans = answer_block.get(quiz_id, [])
            is_correct = (set(user_final_ans) == correct_set)

            # 3.3 汇总所有错误尝试（来自 incorrectAnswers → value 字段）
            incorrect_info = task_info.get("incorrectAnswers", {}) \
                                     .get(quiz_id, {})
            attempts = incorrect_info.get("value", [])

            # 3.4 统计每个选项出现频次
            counter = Counter()
            for attempt in attempts:
                counter.update(attempt)

            # 跳过正确选项，留下纯错误分布
            wrong_choice_distribution = {
                choice: cnt for choice, cnt in counter.items()
                if choice not in correct_set
            }
            wrong_choice_count = sum(wrong_choice_distribution.values())

            quiz_results.append({
                "participantId":             participant_id,
                "format":                    format_name,
                "quiz_key":                  task_key,
                "correct_answer":            correct_answer,
                "user_final_answer":         user_final_ans,
                "correct":                   int(is_correct),   # <- 0 or 1
                "num_wrong_attempts":        len(attempts),
                "all_wrong_attempts_list":   attempts,
                "all_wrong_attempts_frequency": dict(counter),
                "wrong_choice_distribution":     wrong_choice_distribution,
                "wrong_choice_count":            wrong_choice_count
            })

    return pd.DataFrame(quiz_results)



def extract_and_encode_familiarity(
    df_post_survey_format: pd.DataFrame,
    familiarity_mapping: Dict[str,int] = None,
    format_key_map: Dict[str,str]  = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    
    # 1. 默认映射
    if familiarity_mapping is None:
        familiarity_mapping = {
            'Not familiar at all'        : 1,
            'Heard of it but never used' : 2,
            'Used it a few times'        : 3,
            'Comfortable using it'       : 4,
            'Expert'                     : 5
        }
    if format_key_map is None:
        format_key_map = {
            'json' : 'JSON',
            'jsonc': 'JSONC',
            'json5': 'JSON5',
            'hjson': 'HJSON',
            'toml' : 'TOML',
            'xml'  : 'XML',
            'yaml' : 'YAML'
        }

    # 2. 提取并展开 q12
    df_q12 = df_post_survey_format[['participantId','format','q12']].copy()
    df_expanded = pd.json_normalize(df_q12['q12'])
    df_q12_expanded = pd.concat([
        df_q12[['participantId','format']].reset_index(drop=True),
        df_expanded.reset_index(drop=True)
    ], axis=1)
    # 重新排列列顺序
    cols = ['participantId','format'] + [
        c for c in df_q12_expanded.columns
        if c not in ('participantId','format')
    ]
    df_q12_expanded = df_q12_expanded[cols]

    # 3. 对所有格式列做映射编码
    df_encoded = df_q12_expanded.copy()
    for col in format_key_map.values():
        if col in df_encoded.columns:
            df_encoded[col] = df_encoded[col].map(familiarity_mapping)

    # 4. 构建每位 participant 的 familiarity
    df_familiar = df_encoded[['participantId','format']].copy()
    def _lookup(row):
        key = format_key_map.get(row['format'])
        return row.get(key, pd.NA)
    df_familiar['familiarity'] = df_encoded.apply(_lookup, axis=1)

    return df_encoded, df_familiar


def get_answer_counts(
    df: pd.DataFrame,
    question: str,
    group_by: str = None
) -> Union[pd.Series, pd.DataFrame]:
    # 1) 如果该列有多选列表，先 explode
    if df[question].apply(lambda x: isinstance(x, (list, tuple))).any():
        d = df.explode(question)
    else:
        d = df.copy()

    # 2) 丢弃空值
    d = d.dropna(subset=[question])

    # 3) 统计计数
    if group_by:
        counts = (
            d
            .groupby([question, group_by])
            .size()
            .unstack(fill_value=0)
        )
    else:
        counts = d[question].value_counts()

    return counts



def plot_answer_distribution(
    df: pd.DataFrame,
    question: str,
    group_by: str = None,
    figsize: tuple = (8, 5),
    stacked: bool = False
):
    """
    可视化某个问题的答案分布，支持单选和多选（列表）回答。
    如果 group_by 不为 None，会画分组柱状或堆叠柱状图。
    """
    # 提取那一列并丢掉空值
    series = df[question].dropna()

    # 如果检测到列表/元组，先 explode
    if series.apply(lambda x: isinstance(x, (list, tuple))).any():
        df_plot = df.explode(question)
    else:
        df_plot = df

    # 绘图
    if group_by:
        # 分组计数
        counts = (
            df_plot
            .groupby([question, group_by])
            .size()
            .unstack(fill_value=0)
        )
        ax = counts.plot(
            kind='bar',
            stacked=stacked,
            figsize=figsize,
            width=0.8
        )
        ax.set_ylabel('Count')
        ax.set_xlabel(question)
        ax.set_title(f"Distribution of {question} by {group_by}")
        ax.legend(title=group_by, bbox_to_anchor=(1.05,1), loc='upper left')
        plt.xticks(rotation=45, ha='right')
    else:
        # 单总体计数，按频次降序
        counts = df_plot[question].value_counts().sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=figsize)
        ax.bar(counts.index.astype(str), counts.values, width=0.6)
        ax.set_ylabel('Count')
        ax.set_xlabel(question)
        ax.set_title(f"Distribution of {question}")
        plt.xticks(rotation=45, ha='right')

    plt.tight_layout()
    plt.show()


def extract_survey_items(
    df: pd.DataFrame,
    items: list,
    id_col: str = 'participantId',
    format_col: str = 'format'
) -> pd.DataFrame:
    """
    从 df 中提取若干 survey 项，再加上 participantId 和 format。

    参数
    ----
    df         : 原始 DataFrame，需要包含 id_col、format_col，以及 items 列
    items      : 要提取的列名列表，例如 ['q9','q13','q14','q10','q11']
    id_col     : 参与者 ID 列名（默认 'participantId'）
    format_col : 格式列名（默认 'format'）

    返回
    ----
    sub_df : DataFrame，只包含 [id_col, format_col] + items
    """
    # 构造要保留的列列表
    cols = [id_col, format_col] + items
    # 校验一下所有列都存在
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"这些列在 DataFrame 中不存在：{missing}")
    # 返回一个 copy，避免修改原 df
    return df[cols].copy()

def extract_post_task_tlx_responses(all_data: dict, df_part: pd.DataFrame) -> pd.DataFrame:

    records = []
    for pid, rec in all_data.items():
        # 获取 participantId
        participant_id = rec.get('participantId', pid)
        # 获取 TLX answer dict
        survey = rec.get('answers', {}) \
                    .get('post-task-survey-tlx', {}) \
                    .get('answer', {})
        
        row = {'participantId': participant_id}
        # 遍历题目键
        for q_key, q_obj in survey.items():
            # 如果 q_obj 是 dict 且包含 'answer' 字段，就提取它
            if isinstance(q_obj, dict) and 'answer' in q_obj:
                row[q_key] = q_obj['answer']
            else:
                # 否则把整个对象存入
                row[q_key] = q_obj
        records.append(row)
    
    df = pd.DataFrame.from_records(records)
    # 按列排序：participantId 放在最前
    cols = ['participantId'] + [c for c in df.columns if c != 'participantId']
    df_post_survy = df[cols]
    df_post_survy_format = pd.merge(
        df_post_survy,
        df_part[['participantId', 'format']],
        on='participantId',
        how='left'
    )
    bases = ['q9', 'q13', 'q14']
    for base in bases:
        other = f'{base}-other'
        # 如果 other 列不是空（也排除空字符串），就把它的值赋给 base 列
        mask = df_post_survy_format[other].notna() & (df_post_survy_format[other].astype(str).str.strip() != '')
        df_post_survy_format.loc[mask, base] = df_post_survy_format.loc[mask, other]
        # 删除那个 other 列
        df_post_survy_format.drop(columns=[other], inplace=True)

    return df_post_survy_format

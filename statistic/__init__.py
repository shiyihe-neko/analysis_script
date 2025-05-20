import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from scipy.linalg import LinAlgError
from itertools import combinations
from scipy.stats import chi2_contingency, fisher_exact
from statsmodels.stats.multitest import multipletests
from scipy.stats import kruskal, f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import scikit_posthocs as sp


def compare_statistically_orig(df: pd.DataFrame,
                                 group_col: str = 'format',
                                 value_col: str = 'ratio',
                                 method: str = 'kruskal',
                                 task_col: str = None,
                                 discrete_tasks: list = None,
                                 continuous_tasks: list = None) -> dict:
    """
    对多任务(task_col)分别做显著性检验：
      - 若 task 在 discrete_tasks，用 χ² / Fisher 检验
      - 否则若在 continuous_tasks，用 Anova/Kruskal
      - 否则默认用 method
    
    返回：
      单任务 or 多任务(返回 dict of dicts)
    """
    discrete_set = set(discrete_tasks or [])
    continuous_set = set(continuous_tasks or [])

    def _compare_cont(subdf):
        groups = [g[value_col].dropna().values for _, g in subdf.groupby(group_col)]
        res = {'method': method.lower(),
               'overall_p': None,
               'is_significant': False,
               'pairwise': None}
        if method.lower() == 'kruskal':
            stat, p = kruskal(*groups)
            res.update({'overall_p': p, 'is_significant': p < 0.05})
            if p < 0.05:
                d = sp.posthoc_dunn(subdf, val_col=value_col, group_col=group_col, p_adjust='bonferroni')
                rec = []
                for i, j in combinations(d.columns, 2):
                    pv = d.loc[i, j]
                    rec.append({
                        'group1': i, 'group2': j,
                        'p_value': pv,
                        'significant': pv < 0.05,
                        'interpretation': f"{i} vs {j} is {'significant' if pv < 0.05 else 'not significant'}"
                    })
                res['pairwise'] = pd.DataFrame(rec)
        else:
            stat, p = f_oneway(*groups)
            res.update({'overall_p': p, 'is_significant': p < 0.05})
            if p < 0.05:
                tk = pairwise_tukeyhsd(subdf[value_col], subdf[group_col], alpha=0.05)
                tb = pd.DataFrame(tk.summary().data[1:], columns=tk.summary().data[0])
                tb['significant'] = tb['reject']
                tb['interpretation'] = tb.apply(
                    lambda r: f"{r['group1']} vs {r['group2']} is {'significant' if r['reject'] else 'not significant'}",
                    axis=1
                )
                res['pairwise'] = tb
        return res

    def _compare_disc(subdf):
        groups = list(subdf[group_col].dropna().unique())
        table = np.array([
            subdf[subdf[group_col] == g][value_col]
                  .value_counts().reindex([1, 0], fill_value=0).values
            for g in groups
        ])
        chi2, p, _, _ = chi2_contingency(table)
        recs, p_raw = [], []
        for g1, g2 in combinations(groups, 2):
            subtab = table[[groups.index(g1), groups.index(g2)], :]
            try:
                _, p2, _, _ = chi2_contingency(subtab)
            except ValueError:
                _, p2 = fisher_exact(subtab)
            recs.append((g1, g2, p2))
            p_raw.append(p2)
        reject, p_adj, _, _ = multipletests(p_raw, alpha=0.05, method='bonferroni')
        records = []
        for (g1, g2, p2), padj, sig in zip(recs, p_adj, reject):
            records.append({
                'group1': g1, 'group2': g2,
                'p_raw': p2, 'p_adj': padj,
                'significant': bool(sig),
                'interpretation': f"{g1} vs {g2} is {'significant' if sig else 'not significant'}"
            })
        return {'method': 'chi2/fisher',
                'overall_p': p,
                'is_significant': p < 0.05,
                'pairwise': pd.DataFrame(records)}

    def runner(subdf, task):
        if task in discrete_set:
            return _compare_disc(subdf)
        if continuous_set:
            if task in continuous_set:
                return _compare_cont(subdf)
            else:
                return _compare_cont(subdf)
        return _compare_cont(subdf)

    if task_col and df[task_col].nunique() > 1:
        out = {}
        for t in df[task_col].dropna().unique():
            sub = df[df[task_col] == t]
            out[t] = runner(sub, t)
        return out
    return runner(df, None)


# def compare_statistically(df: pd.DataFrame,
#                                  group_col: str = 'format',
#                                  value_col: str = 'ratio',
#                                  method: str = 'kruskal',
#                                  task_col: str = None,
#                                  discrete_tasks: list = None,
#                                  continuous_tasks: list = None) -> dict:
#     discrete_set = set(discrete_tasks or [])
#     continuous_set = set(continuous_tasks or [])

#     def _compare_cont(subdf):
#         groups = [g[value_col].dropna().values for _, g in subdf.groupby(group_col)]
#         res = {
#             'method': method.lower(),
#             'overall_p': None,
#             'is_significant': False,
#             'pairwise': pd.DataFrame(columns=['group1','group2','p_value','significant','interpretation'])
#         }
#         if method.lower() == 'kruskal':
#             _, p = kruskal(*groups)
#             res['overall_p'], res['is_significant'] = p, (p < 0.05)
#             if p < 0.05:
#                 d = sp.posthoc_dunn(subdf, val_col=value_col, group_col=group_col, p_adjust='bonferroni')
#                 recs = []
#                 for i, j in combinations(d.columns, 2):
#                     pv = d.loc[i, j]
#                     recs.append({
#                         'group1': i, 'group2': j,
#                         'p_value': pv,
#                         'significant': pv < 0.05,
#                         'interpretation': f"{i} vs {j} is {'significant' if pv < 0.05 else 'not significant'}"
#                     })
#                 res['pairwise'] = pd.DataFrame(recs)
#         else:  # anova
#             _, p = f_oneway(*groups)
#             res['overall_p'], res['is_significant'] = p, (p < 0.05)
#             if p < 0.05:
#                 tk = pairwise_tukeyhsd(subdf[value_col], subdf[group_col], alpha=0.05)
#                 tb = pd.DataFrame(tk.summary().data[1:], columns=tk.summary().data[0])
#                 tb = tb.rename(columns={'p-adj':'p_value'})
#                 tb['significant'] = tb['reject']
#                 tb['interpretation'] = tb.apply(
#                     lambda r: f"{r['group1']} vs {r['group2']} is "
#                               f"{'significant' if r['reject'] else 'not significant'}",
#                     axis=1
#                 )
#                 res['pairwise'] = tb[['group1','group2','p_value','significant','interpretation']]
#         return res


    # def _compare_disc(subdf):
    #     groups = list(subdf[group_col].dropna().unique())
    #     table = np.array([
    #         subdf[subdf[group_col] == g][value_col]
    #               .value_counts().reindex([1, 0], fill_value=0).values
    #         for g in groups
    #     ])
    #     chi2, p, _, _ = chi2_contingency(table)
    #     recs, p_raw = [], []
    #     for g1, g2 in combinations(groups, 2):
    #         subtab = table[[groups.index(g1), groups.index(g2)], :]
    #         try:
    #             _, p2, _, _ = chi2_contingency(subtab)
    #         except ValueError:
    #             _, p2 = fisher_exact(subtab)
    #         recs.append((g1, g2, p2))
    #         p_raw.append(p2)
    #     reject, p_adj, _, _ = multipletests(p_raw, alpha=0.05, method='bonferroni')
    #     records = []
    #     for (g1, g2, _), padj, sig in zip(recs, p_adj, reject):
    #         records.append({
    #             'group1': g1, 'group2': g2,
    #             'p_value': padj,
    #             'significant': bool(sig),
    #             'interpretation': f"{g1} vs {g2} is {'significant' if sig else 'not significant'}"
    #         })
    #     return {
    #         'method': 'chi2/fisher',
    #         'overall_p': p,
    #         'is_significant': p < 0.05,
    #         'pairwise': pd.DataFrame(records, columns=['group1','group2','p_value','significant','interpretation'])
    #     }

    # def runner(subdf, task):
    #     if task in discrete_set:
    #         return _compare_disc(subdf)
    #     if continuous_set and task in continuous_set:
    #         return _compare_cont(subdf)
    #     # 默认
    #     return _compare_cont(subdf)

    # if task_col and df[task_col].nunique() > 1:
    #     out = {}
    #     for t in df[task_col].dropna().unique():
    #         sub = df[df[task_col] == t]
    #         out[t] = runner(sub, t)
    #     return out
    # return runner(df, None)

def compare_statistically(
    df: pd.DataFrame,
    group_col: str = 'format',
    value_col: str = 'ratio',
    method: str = 'kruskal',
    p_adjust_method: str = 'bonferroni',
    task_col: str = None,
    discrete_tasks: list = None,
    continuous_tasks: list = None,
    alpha: float = 0.05
) -> dict:
    """
    对 df 按 group_col 做组间比较，可选 Kruskal-Wallis 或 ANOVA，
    并返回：
      - descriptive: 各组的描述性统计（count, mean, std, median, min, max）
      - overall_p:    全局检验的 p 值
      - effect_size： 效应量（ε² 或 η²）
      - is_significant: 全局检验是否显著
      - pairwise:     事后两两比较结果表
      - warning:      全局显著但无事后显著时提醒
      - method, alpha, p_adjust_method: 元数据

    如果指定了 task_col 且含多个任务，则对每个任务分别做比较，返回一个 dict。
    """
    discrete_set = set(discrete_tasks or [])
    continuous_set = set(continuous_tasks or [])

    def _compare_cont(subdf: pd.DataFrame) -> dict:
        # 1) 清洗并计算描述性统计
        data = subdf.dropna(subset=[value_col, group_col]).copy()
        desc = data.groupby(group_col)[value_col] \
                   .agg(['count','mean','std','median','min','max'])
        
        # 2) 分组数据列表
        groups = [g[value_col].values for _, g in data.groupby(group_col)]
        
        # 3) 初始化结果
        res = {
            'method': method.lower(),
            'alpha': alpha,
            'p_adjust_method': p_adjust_method,
            'descriptive': desc,
            'overall_p': np.nan,
            'effect_size': np.nan,
            'is_significant': False,
            'pairwise': pd.DataFrame(columns=[
                'group1','group2','p_value','significant','interpretation'
            ]),
            'warning': None
        }

        # 4) 样本量检查
        if len(groups) < 2:
            res['warning'] = 'Not enough groups for statistical test'
            return res

        # 5) 全局检验 + 效应量
        if method.lower() == 'kruskal':
            H, p = kruskal(*groups)
            res['overall_p'] = p
            res['is_significant'] = (p < alpha)
            # ε² = (H − k + 1)/(n − k)
            k = len(groups)
            n = data[value_col].shape[0]
            res['effect_size'] = ((H - k + 1) / (n - k)
                                  if n > k else np.nan)
            # 事后多重比较：Dunn
            if p < alpha:
                d = sp.posthoc_dunn(
                    data, val_col=value_col, group_col=group_col,
                    p_adjust=p_adjust_method
                )
                recs = []
                for i, j in combinations(d.columns, 2):
                    pv = d.loc[i, j]
                    sig = pv < alpha
                    recs.append({
                        'group1': i,
                        'group2': j,
                        'p_value': pv,
                        'significant': sig,
                        'interpretation':
                            f"{i} vs {j} is "
                            f"{'significant' if sig else 'not significant'}"
                    })
                res['pairwise'] = pd.DataFrame(recs)

        else:  # ANOVA + η²
            f_stat, p = f_oneway(*groups)
            res['overall_p'] = p
            res['is_significant'] = (p < alpha)
            overall_mean = data[value_col].mean()
            ss_between = sum(
                len(g[value_col]) * (g[value_col].mean() - overall_mean)**2
                for _, g in data.groupby(group_col)
            )
            ss_total = ((data[value_col] - overall_mean)**2).sum()
            res['effect_size'] = (ss_between / ss_total
                                  if ss_total > 0 else np.nan)
            # 事后多重比较：Tukey HSD
            if p < alpha:
                tk = pairwise_tukeyhsd(
                    data[value_col], data[group_col], alpha=alpha
                )
                tb = pd.DataFrame(tk.summary().data[1:],
                                  columns=tk.summary().data[0])
                tb = tb.rename(columns={'p-adj':'p_value'})
                tb['significant'] = tb['reject']
                tb['interpretation'] = tb.apply(
                    lambda r: (
                        f"{r['group1']} vs {r['group2']} is "
                        f"{'significant' if r['reject'] else 'not significant'}"
                    ),
                    axis=1
                )
                res['pairwise'] = tb[[
                    'group1','group2','p_value','significant','interpretation'
                ]]

        # 6) 全局显著但无事后显著时警告
        if (res['overall_p'] < alpha and
            not res['pairwise']['significant'].any()):
            res['warning'] = (
                "Global test significant but no pairwise comparison "
                "passed alpha; consider effect sizes or larger sample"
            )
        return res

    # 外层：按任务拆分
    if task_col and df[task_col].nunique() > 1:
        out = {}
        for t, sub in df.groupby(task_col):
            out[t] = _compare_cont(sub)
        return out
    else:
        return _compare_cont(df)

def vis_violin_comparison(df: pd.DataFrame,
                                group_col: str = 'format',
                                value_col: str = 'ratio',
                                method: str = 'anova',
                                show_significance: bool = True,
                                task_col: str = None,
                                discrete_tasks: list = None,
                                continuous_tasks: list = None):
    discrete_set = set(discrete_tasks or [])
    continuous_set = set(continuous_tasks or [])

    def _plot(subdf, title_suffix=None):
        formats = sorted(subdf[group_col].dropna().unique())
        res = compare_statistically(
            subdf, group_col, value_col, method,
            None, discrete_tasks, continuous_tasks
        )
        sig_df = res['pairwise']
        # sig_df guaranteed DataFrame
        sig_mat = pd.DataFrame('', index=formats, columns=formats)
        for _, r in sig_df.iterrows():
            if r['significant']:
                sig_mat.loc[r['group1'], r['group2']] = '*'
                sig_mat.loc[r['group2'], r['group1']] = '*'

        fig, (axm, axv) = plt.subplots(1, 2, figsize=(14, 6),
                                       gridspec_kw={'width_ratios': [1.2, 2]})
        if title_suffix: fig.suptitle(title_suffix, y=1.02)

        sns.heatmap(sig_mat.isin(['*']), annot=sig_mat, fmt='',
                    cbar=False, cmap='Reds', linewidths=1,
                    linecolor='gray', ax=axm)
        axm.set_xticklabels(formats, rotation=45)
        axm.set_yticklabels(formats, rotation=0)
        axm.set_title('Pairwise Significance')

        sns.violinplot(y=group_col, x=value_col, data=subdf,
                       order=formats, inner='box',
                       scale='width', cut=0, linewidth=1, ax=axv)
        sns.stripplot(y=group_col, x=value_col, data=subdf,
                      order=formats, color='black',
                      alpha=0.3, jitter=True, ax=axv)
        axv.set_title('Distribution by Format (Violin + Box)')
        axv.set_xlabel(value_col)
        axv.set_ylabel(group_col)
        plt.tight_layout()
        plt.show()

    if task_col and df[task_col].nunique() > 1:
        for t in df[task_col].dropna().unique():
            _plot(df[df[task_col] == t], title_suffix=f"Task: {t}")
    else:
        _plot(df)



def vis_box_comparasion(df: pd.DataFrame,
                               group_col: str = 'format',
                               value_col: str = 'ratio',
                               method: str = 'anova',
                               show_significance: bool = True,
                               box_height: float = 0.3,
                               vpad: float = 1.5,
                               wspace: float = 0.4,
                               palette_name: str = 'tab10',
                               task_col: str = None,
                               discrete_tasks: list = None,
                               continuous_tasks: list = None):
    """
    对比箱形图 + 半 KDE，可捕获 GaussianKDE 在退化数据下的异常。
    """
    discrete_set = set(discrete_tasks or [])
    continuous_set = set(continuous_tasks or [])

    def _plot(subdf, title_suffix=None):
        formats = subdf[group_col].dropna().unique().tolist()
        n = len(formats)

        # 1) 生成显著性矩阵（略，用 compare_groups_statistically）
        res = compare_statistically(subdf, group_col, value_col, method,
                                           None, discrete_tasks, continuous_tasks)
        sig_df = res['pairwise']
        sig_mat = pd.DataFrame('', index=formats, columns=formats)
        for _, r in sig_df.iterrows():
            if r['significant']:
                sig_mat.loc[r['group1'], r['group2']] = '*'
                sig_mat.loc[r['group2'], r['group1']] = '*'

        # 2) 画布和热力图
        palette = sns.color_palette(palette_name, n)
        cmap = {fmt: palette[i] for i, fmt in enumerate(formats)}
        fig, (axm, axb) = plt.subplots(1, 2, figsize=(14, 6),
                                       gridspec_kw={'width_ratios': [1.2, 2]})
        fig.subplots_adjust(wspace=wspace)
        if title_suffix:
            fig.suptitle(title_suffix, y=1.02)

        sns.heatmap(sig_mat.isin(['*']), annot=sig_mat, fmt='',
                    cbar=False, cmap='Reds', linewidths=1,
                    linecolor='gray', ax=axm)
        axm.set_xticklabels(formats, rotation=45)
        axm.set_yticklabels(formats, rotation=0)
        axm.set_title('Pairwise Significance')

        # 3) 箱形图 + 散点
        axb.set_axisbelow(False)
        y_pos = {fmt: (n - 1 - idx) * vpad for idx, fmt in enumerate(formats)}
        half = box_height / 2

        for fmt in formats:
            data = subdf.loc[subdf[group_col] == fmt, value_col].dropna().values
            y = y_pos[fmt]
            if data.size < 1:
                continue

            # 箱形图
            axb.boxplot(data, positions=[y], vert=False, widths=box_height,
                        patch_artist=True,
                        boxprops=dict(facecolor=cmap[fmt], alpha=0.6),
                        whiskerprops=dict(color=cmap[fmt], linewidth=1),
                        medianprops=dict(color='black'),
                        flierprops=dict(marker='o', color=cmap[fmt], alpha=0.6))

            # 散点
            axb.scatter(data, np.full_like(data, y),
                        color=cmap[fmt], alpha=0.6, s=10, zorder=5)

            # 半 KDE 曲线：捕获退化异常
            if data.size >= 2:
                try:
                    kde = gaussian_kde(data)
                    xs = np.linspace(data.min(), data.max(), 200)
                    dens = kde(xs)
                    dens = dens / dens.max() * half
                    axb.plot(xs, y + half + dens,
                             color=cmap[fmt], linewidth=2, alpha=0.8,
                             zorder=10, clip_on=False)
                except LinAlgError:
                    # 数据退化时跳过 KDE
                    pass

        # 4) 坐标和标签
        axb.set_yticks(list(y_pos.values()))
        axb.set_yticklabels(formats)
        axb.set_title('Box + Density per Format')
        axb.set_xlabel(value_col)
        axb.set_ylabel(group_col)

        plt.tight_layout()
        plt.show()

    # 如果有多个 task，就按 task 分图
    if task_col and df[task_col].nunique() > 1:
        for t in df[task_col].dropna().unique():
            _plot(df[df[task_col] == t], title_suffix=f"Task: {t}")
    else:
        _plot(df)


def compare_significant_pairs(df: pd.DataFrame,
                              group_col: str = 'format',
                              value_col: str = 'ratio',
                              method: str = 'kruskal',
                              task_col: str = None,
                              discrete_tasks: list = None,
                              continuous_tasks: list = None) -> dict:
    """
    只保留显著性两两比较的结果：
      - 对每个 task（或整体）调用 compare_statistically
      - 过滤 pairwise DataFrame，只保留 `significant == True` 的行
      - 如果整体不显著，返回空 DataFrame

    返回：
      如果 task_col 不传或只有一个任务，返回单个 DataFrame；
      否则返回 { task1: df1, task2: df2, … }
    """
    # 复用之前已定义的 compare_statistically 函数
    def _filter(res: dict):
        # 如果整体不显著，直接返回空表
        if not res['is_significant']:
            return pd.DataFrame(columns=['group1','group2','p_value','significant','interpretation'])
        # 否则从 pairwise 中筛出 significant == True
        df_pw = res['pairwise']
        return df_pw[df_pw['significant'] == True].reset_index(drop=True)

    # 多任务模式
    if task_col and df[task_col].nunique() > 1:
        out = {}
        for t in df[task_col].dropna().unique():
            sub = df[df[task_col] == t]
            res = compare_statistically(
                sub, group_col, value_col, method,
                None, discrete_tasks, continuous_tasks
            )
            out[t] = _filter(res)
        return out

    # 单任务模式
    res = compare_statistically(
        df, group_col, value_col, method,
        None, discrete_tasks, continuous_tasks
    )
    return _filter(res)

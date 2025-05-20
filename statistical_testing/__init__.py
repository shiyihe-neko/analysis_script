import pandas as pd
from scipy.stats import kruskal, f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import scikit_posthocs as sp
from typing import Tuple
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from itertools import combinations
from scipy.stats import gaussian_kde



def compare_groups_statistically(df: pd.DataFrame,
                                 group_col: str = 'format',
                                 value_col: str = 'ratio',
                                 method: str = 'kruskal',
                                 task_col: str = None) -> dict:
    """
    对一列或多任务（task_col）内的各 format 分组做显著性检验。

    如果 task_col 不为空且包含 >1 个不同值，
    则对每个任务分别返回一个检验结果字典，最终返回 {task: result_dict}。
    否则直接返回单个 result_dict。

    单个 result_dict 字段：
      - method: 用的方法
      - overall_p: 整体检验 p 值
      - is_significant: 整体是否显著 (p<0.05)
      - pairwise: 如果整体显著，两两比较的 DataFrame
    """
    def _compare(subdf: pd.DataFrame) -> dict:
        grouped = [g[value_col].dropna().values for _, g in subdf.groupby(group_col)]
        res = {'method': method.lower(),
               'overall_p': None,
               'is_significant': False,
               'pairwise': None}

        if method.lower() == 'kruskal':
            stat, p = kruskal(*grouped)
            res['overall_p'] = p
            res['is_significant'] = (p < 0.05)
            if p < 0.05:
                dunn = sp.posthoc_dunn(subdf,
                                       val_col=value_col,
                                       group_col=group_col,
                                       p_adjust='bonferroni')
                records = []
                for i,j in combinations(dunn.columns, 2):
                    pv = dunn.loc[i,j]
                    records.append({
                        'group1': i,
                        'group2': j,
                        'p_value': pv,
                        'significant': (pv<0.05),
                        'interpretation': f"{i} vs {j} is {'significant' if pv<0.05 else 'not significant'}"
                    })
                res['pairwise'] = pd.DataFrame(records)

        elif method.lower() == 'anova':
            stat, p = f_oneway(*grouped)
            res['overall_p'] = p
            res['is_significant'] = (p < 0.05)
            if p < 0.05:
                tukey = pairwise_tukeyhsd(subdf[value_col],
                                          subdf[group_col],
                                          alpha=0.05)
                tukey_df = pd.DataFrame(tukey.summary().data[1:],
                                        columns=tukey.summary().data[0])
                tukey_df['significant'] = tukey_df['reject']
                tukey_df['interpretation'] = tukey_df.apply(
                    lambda r: f"{r['group1']} vs {r['group2']} is "
                              f"{'significant' if r['reject'] else 'not significant'}",
                    axis=1
                )
                res['pairwise'] = tukey_df

        else:
            raise ValueError("method must be 'kruskal' or 'anova'")

        return res

    # 如果指定了 task_col 且有多个任务，则分别计算并返回 dict
    if task_col and df[task_col].nunique() > 1:
        output = {}
        for task in df[task_col].dropna().unique():
            sub = df[df[task_col] == task]
            output[task] = _compare(sub)
        return output

    # 否则只做一次
    return _compare(df)


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from itertools import combinations
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import scikit_posthocs as sp

def visualize_violin_comparison(df: pd.DataFrame,
                                group_col: str = 'format',
                                value_col: str = 'ratio',
                                method: str = 'anova',
                                show_significance: bool = True,
                                task_col: str = None):
    """
    左：格式之间的显著性比较
    右：violin plot（横向分布+箱线）

    新增：
      • task_col: 如果传入且有多个任务值，就对每个任务分别绘一张图
    """
    def _plot(subdf: pd.DataFrame, title_suffix: str = None):
        # 1. 固定顺序
        formats = sorted(subdf[group_col].dropna().unique())

        # 2. 构建显著性矩阵
        sig = pd.DataFrame('', index=formats, columns=formats)
        if show_significance:
            if method == 'anova':
                tukey = pairwise_tukeyhsd(endog=subdf[value_col],
                                          groups=subdf[group_col],
                                          alpha=0.05)
                tukey_df = pd.DataFrame(tukey.summary().data[1:], 
                                        columns=tukey.summary().data[0])
                for _, row in tukey_df.iterrows():
                    g1, g2 = row['group1'], row['group2']
                    mark = '*' if bool(row['reject']) else ''
                    sig.loc[g1, g2] = mark
                    sig.loc[g2, g1] = mark

            else:  # kruskal
                dunn = sp.posthoc_dunn(subdf, val_col=value_col,
                                       group_col=group_col,
                                       p_adjust='bonferroni')
                for i, j in combinations(dunn.columns, 2):
                    p = dunn.loc[i, j]
                    mark = '*' if p < 0.05 else ''
                    sig.loc[i, j] = mark
                    sig.loc[j, i] = mark

        # 3. 画布
        fig, (ax_mat, ax_vio) = plt.subplots(1, 2, figsize=(14, 6),
                                             gridspec_kw={'width_ratios': [1.2, 2]})
        if title_suffix:
            fig.suptitle(title_suffix, y=1.02, fontsize=16)

        # 左：热力图
        sns.heatmap(sig.isin(['*']), annot=sig, fmt='',
                    cbar=False, cmap='Reds',
                    linewidths=1, linecolor='gray', ax=ax_mat)
        ax_mat.set_title('Pairwise Significance')
        ax_mat.set_xticklabels(formats, rotation=45)
        ax_mat.set_yticklabels(formats, rotation=0)

        # 右：Violin + Box + Strip
        sns.violinplot(y=group_col, x=value_col, data=subdf,
                       order=formats, inner='box',
                       scale='width', cut=0, linewidth=1,
                       ax=ax_vio)
        sns.stripplot(y=group_col, x=value_col, data=subdf,
                      order=formats, color='black',
                      alpha=0.3, jitter=True, ax=ax_vio)

        ax_vio.set_title('Distribution by Format (Violin + Box)')
        ax_vio.set_xlabel(value_col)
        ax_vio.set_ylabel(group_col)

        plt.tight_layout()
        plt.show()

    # 如果没有 task_col 或者只有一个任务，直接绘制一次
    if not task_col or df[task_col].nunique() <= 1:
        _plot(df)
    else:
        # 对每个 task 分开绘图
        for t in df[task_col].dropna().unique():
            sub = df[df[task_col] == t]
            _plot(sub, title_suffix=f"Task: {t}")




def visualize_box_with_density(df: pd.DataFrame,
                               group_col: str = 'format',
                               value_col: str = 'ratio',
                               method: str = 'anova',
                               show_significance: bool = True,
                               box_height: float = 0.3,
                               vpad: float = 1.5,
                               wspace: float = 0.4,
                               palette_name: str = 'tab10',
                               task_col: str = None):
    """
    左：显著性矩阵
    右：每组横向 boxplot+scatter+上半KDE（箱体上色）

    新增参数：
      • task_col: 如果传入，对 df[task_col] 中每个唯一值分别绘制一张图；
                  否则把整个 df 当一个整体来画。
      • box_height, vpad, wspace, palette_name 同之前。
    """
    def _plot(subdf: pd.DataFrame, title_suffix: str = None):
        # 1. formats 顺序
        formats = subdf[group_col].dropna().unique().tolist()
        n = len(formats)

        # 2. 显著性矩阵
        sig = pd.DataFrame('', index=formats, columns=formats)
        if show_significance:
            if method == 'anova':
                tukey = pairwise_tukeyhsd(subdf[value_col], subdf[group_col], alpha=0.05)
                tdf = pd.DataFrame(tukey.summary().data[1:], columns=tukey.summary().data[0])
                for _, r in tdf.iterrows():
                    g1,g2 = r['group1'], r['group2']
                    mark = '*' if bool(r['reject']) else '×'
                    if g1 in sig.index and g2 in sig.columns:
                        sig.loc[g1,g2] = mark; sig.loc[g2,g1] = mark
            else:
                dunn = sp.posthoc_dunn(subdf, val_col=value_col, group_col=group_col, p_adjust='bonferroni')
                for i,j in combinations(dunn.columns,2):
                    if i in sig.index and j in sig.columns:
                        mark = '*' if dunn.loc[i,j] < 0.05 else '×'
                        sig.loc[i,j] = mark; sig.loc[j,i] = mark

        # 3. 配色
        palette = sns.color_palette(palette_name, n)
        color_map = {fmt: palette[i] for i, fmt in enumerate(formats)}

        # 4. 画布
        fig, (ax_sig, ax_box) = plt.subplots(1,2,figsize=(14,6),
                                             gridspec_kw={'width_ratios':[1.2,2]})
        fig.subplots_adjust(wspace=wspace)
        if title_suffix:
            fig.suptitle(title_suffix, y=1.02, fontsize=16)

        # left heatmap
        sns.heatmap(sig.isin(['*']), annot=sig, fmt='',
                    cbar=False, cmap='Reds', linewidths=1, linecolor='gray',
                    ax=ax_sig)
        ax_sig.set_title('Pairwise Significance')
        ax_sig.set_xticklabels(formats, rotation=45)
        ax_sig.set_yticklabels(formats, rotation=0)

        # right box+scatter+KDE
        ax_box.set_axisbelow(False)
        y_pos = {fmt: (n-1-idx)*vpad for idx, fmt in enumerate(formats)}
        half = box_height/2

        # box
        for fmt in formats:
            data = subdf.loc[subdf[group_col]==fmt, value_col].dropna().values
            y = y_pos[fmt]
            if data.size<1: continue
            ax_box.boxplot(data, positions=[y], vert=False, widths=box_height,
                           patch_artist=True,
                           boxprops=dict(facecolor=color_map[fmt], alpha=0.6),
                           whiskerprops=dict(color=color_map[fmt], linewidth=1),
                           medianprops=dict(color='black'),
                           capprops=dict(color=color_map[fmt]),
                           flierprops=dict(marker='o', color=color_map[fmt], alpha=0.6))

        # scatter
        for fmt in formats:
            data = subdf.loc[subdf[group_col]==fmt, value_col].dropna().values
            y = y_pos[fmt]
            ax_box.scatter(data, np.full_like(data, y),
                           color=color_map[fmt], alpha=0.6, s=10, zorder=5)

        # upper KDE
        for fmt in formats:
            data = subdf.loc[subdf[group_col]==fmt, value_col].dropna().values
            y = y_pos[fmt]
            if data.size<2: continue
            kde = gaussian_kde(data)
            xs = np.linspace(data.min(), data.max(), 200)
            dens = kde(xs)
            dens = dens/dens.max()*half
            ax_box.plot(xs, y+half+dens,
                        color=color_map[fmt], linewidth=2, alpha=0.8,
                        zorder=10, clip_on=False)

        ax_box.set_yticks(list(y_pos.values()))
        ax_box.set_yticklabels(formats)
        ax_box.set_title('Box + Density per Format')
        ax_box.set_xlabel(value_col)
        ax_box.set_ylabel(group_col)
        plt.tight_layout()
        plt.show()


    # 如果没传 task_col，或者只有一个 task，直接画一次
    if not task_col or df[task_col].nunique() <= 1:
        _plot(df)
    else:
        # 对每个 task 分别作图
        for t in df[task_col].dropna().unique().tolist():
            subdf = df[df[task_col] == t]
            _plot(subdf, title_suffix=f"Task: {t}")
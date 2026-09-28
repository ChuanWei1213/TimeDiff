"""Recompute saved SPX experiments and export figures for the Notion report; no training."""
from pathlib import Path
import json
import hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/spx_three_way_20260929'
RUNS = {
    'Mixup': ('outputs/spx/crash_20260929/mixup_samples', 'outputs/spx/initial'),
    'No mixup': ('outputs/spx/crash_20260929/no_mixup_samples', 'outputs/spx/crash_20260929/train'),
    'No mixup + loss': ('outputs/spx/condition_lambda_0p1/samples', 'outputs/spx/condition_lambda_0p1/train'),
}
COLORS = ['#89929B', '#3976B8', '#D68A27']


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    summaries, paths, manifest = [], {}, {}
    reference = None
    for method, (sample, train) in RUNS.items():
        sample, train = ROOT / sample, ROOT / train
        info = json.loads((sample / 'sampling.json').read_text())
        meta = json.loads((train / 'metadata.json').read_text())
        shared = {k: info[k] for k in ['history_end', 'seed', 'n_paths', 'conditions', 'data_sha256', 'return_tolerance', 'volatility_tolerance', 'batch_size']}
        if reference is None:
            reference = shared
        assert shared == reference
        p = pd.read_csv(sample / 'paths.csv')
        assert len(p) == 7 * 200 * 21 and not p.duplicated(['scenario', 'path', 'step']).any()
        grouped = p.groupby(['scenario', 'path'])
        assert (grouped.size() == 21).all()
        actual = grouped.rclose.agg(return_sum='sum', std='std')
        actual['return'] = np.expm1(actual.return_sum)
        actual['volatility'] = actual['std'] * np.sqrt(252)
        actual = actual.reset_index()
        saved = pd.read_csv(sample / 'path_metrics.csv').sort_values(['scenario', 'path'])
        np.testing.assert_allclose(actual[['return','volatility']], saved[['return','volatility']], atol=1e-10)
        rows = []
        for scenario, (target_r, target_v) in enumerate(info['conditions']):
            a = actual[actual.scenario == scenario]
            raw = p[p.scenario == scenario]
            hit = (abs(a['return'] - target_r) <= .02) & (abs(a.volatility - target_v) <= .05)
            rows.append(dict(scenario=scenario, target_return=target_r, target_volatility=target_v,
                return_mean=a['return'].mean(), volatility_mean=a.volatility.mean(),
                return_mae=abs(a['return']-target_r).mean(), volatility_mae=abs(a.volatility-target_v).mean(),
                joint_hit_rate=hit.mean(), rhigh_clip_rate=(raw.rhigh_raw < 0).mean(), rlow_clip_rate=(raw.rlow_raw > 0).mean()))
        result = pd.DataFrame(rows)
        saved_summary = pd.read_csv(sample / 'summary.csv')
        np.testing.assert_allclose(result, saved_summary[result.columns], atol=1e-10)
        result.insert(0, 'method', method)
        summaries.append(result)
        paths[method] = actual
        manifest[method] = dict(sample=str(sample.relative_to(ROOT)), train=str(train.relative_to(ROOT)),
            paths_sha256=hashlib.sha256((sample/'paths.csv').read_bytes()).hexdigest(),
            best_step=meta['best_step'], best_validation_condition_mse=meta['best_validation_condition_mse'])
    summary = pd.concat(summaries, ignore_index=True)
    summary.to_csv(OUT / 'three_way_summary.csv', index=False)
    metrics = ['return_mae','volatility_mae','joint_hit_rate','rhigh_clip_rate','rlow_clip_rate']
    aggregate = summary.groupby('method', sort=False)[metrics].mean()
    aggregate.to_csv(OUT / 'aggregate.csv')
    crash = summary[summary.scenario >= 4].groupby('method', sort=False)[metrics].mean()
    crash.to_csv(OUT / 'crash_aggregate.csv')
    train_conditions = pd.read_csv(ROOT / 'outputs/spx/crash_20260929/train/train_conditions.csv')
    support = [int(((abs(train_conditions['return']-r)<=.02)&(abs(train_conditions.volatility-v)<=.05)).sum()) for r,v in reference['conditions']]
    manifest.update(common=reference, support_windows=support, validation='All 4200 paths recomputed; all saved metrics match within 1e-10.')
    (OUT / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n')
    plt.rcParams.update({'font.family':['Arial Unicode MS', 'Heiti TC', 'sans-serif'], 'font.size':12, 'axes.unicode_minus':False, 'svg.fonttype':'none',
        'axes.spines.top':False, 'axes.spines.right':False, 'axes.edgecolor':'#AAB0B6', 'text.color':'#243040', 'axes.labelcolor':'#243040'})
    names = ['Mixup','No mixup','No mixup\n+ loss (λ=0.1)']
    fig, axes = plt.subplots(1,3,figsize=(13.5,4.7))
    for ax, metric, title, unit in zip(axes, metrics[:3], ['報酬平均絕對誤差 ↓','波動率平均絕對誤差 ↓','雙條件命中率 ↑'], ['百分點','百分點','%']):
        values = aggregate[metric].to_numpy()*100
        bars=ax.bar(names, values, color=COLORS, width=.58)
        for b,v in zip(bars,values):
            ax.text(b.get_x()+b.get_width()/2, v+max(values)*.025, f'{v:.2f}', ha='center', fontsize=13)
        ax.set_ylim(0,max(values)*1.23)
        ax.set_title(title, loc='left', pad=17, fontsize=15)
        ax.set_ylabel(unit)
        ax.yaxis.grid(True, alpha=.15)
        ax.set_axisbelow(True)
        ax.tick_params(axis='x', length=0)
    fig.suptitle('三組實驗比較：移除 mixup，再加入條件一致性 loss', x=.04, ha='left', fontsize=18)
    fig.text(.04,.018,'SPX｜同一 63 天歷史｜7 組條件 × 每組 200 條 21 天路徑｜七組等權；非 RL 績效',fontsize=11,color='#5C6671')
    fig.tight_layout(rect=[0,.075,1,.90], w_pad=2)
    fig.savefig(OUT/'overview.png', dpi=180)
    fig.savefig(OUT/'overview.svg')
    plt.close(fig)
    fig, axes = plt.subplots(2,3,figsize=(13.5,7.7))
    for col, scenario in enumerate([4,5,6]):
        target_r,target_v=reference['conditions'][scenario]
        for row,(field,target,tol,limits) in enumerate([('return',target_r,.02,(-60,35)),('volatility',target_v,.05,(0,120))]):
            ax=axes[row,col]
            data=[paths[m].loc[paths[m].scenario==scenario,field].to_numpy()*100 for m in RUNS]
            box=ax.boxplot(data,patch_artist=True,widths=.48,whis=(5,95),showfliers=False,
                medianprops={'color':'#202B36','linewidth':1.5})
            for patch,color in zip(box['boxes'],COLORS): patch.set_facecolor(color); patch.set_alpha(.8)
            ax.axhspan((target-tol)*100,(target+tol)*100,color='#D8DEE5',alpha=.38)
            ax.axhline(target*100,color='#243040',ls='--',lw=1.4)
            ax.text(.02,target*100, f'目標 {target:.0%}',transform=ax.get_yaxis_transform(),ha='left',va='bottom',fontsize=11,
                bbox={'facecolor':'white','edgecolor':'none','alpha':.9,'pad':1})
            ax.set_ylim(*limits); ax.set_xticks([1,2,3], ['Mixup','No mixup','+ loss'])
            ax.yaxis.grid(True,alpha=.15); ax.set_axisbelow(True)
            ax.set_ylabel('21 天累積報酬 (%)' if row==0 else '年化波動率 (%)')
            if row==0: ax.set_title(f'情境：報酬 {target_r:.0%}／波動 {target_v:.0%}',pad=15,fontsize=14)
    fig.suptitle('極端情境的分布：控制改善，但最嚴重跌幅仍偏離目標',x=.04,ha='left',fontsize=18)
    fig.text(.04,.018,'每組 200 條；箱體 P25–P75、中線中位數、鬚 P5–P95（非信賴區間）；虛線為目標，灰帶為命中容許範圍。',fontsize=11,color='#5C6671')
    fig.tight_layout(rect=[0,.065,1,.92], w_pad=2,h_pad=2)
    fig.savefig(OUT/'crash_distributions.png',dpi=180)
    fig.savefig(OUT/'crash_distributions.svg')
    plt.close(fig)
    for svg in OUT.glob('*.svg'):
        svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
    print('All raw path checks passed.\n',aggregate.to_string(), '\nCrash:\n',crash.to_string(), '\nSupport:',support)


if __name__ == '__main__':
    main()

"""Read-only diagnostics of the frozen batch; write only into this new directory."""
from pathlib import Path
import hashlib
import json
import sys

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'output_layout.py').is_file())
sys.path.insert(0, str(ROOT))
from output_layout import JOINT_RUN, resolve_output_path, verify_historical_hash
from data_provider.dow10_cond import load_features, COLUMNS, CLOSE_INDICES, split_windows

BATCH = ROOT / 'outputs/dow10/conditional/empirical_joint/run_20261017/batches/full70_tail30_pr30'
TRAIN = JOINT_RUN / 'train'


def sha(path):
    return hashlib.sha256(resolve_output_path(path).read_bytes()).hexdigest()


def stats(raw):
    raw = np.asarray(raw, dtype=np.float64)
    r = raw[..., ::3]
    center = r - r.mean(axis=1, keepdims=True)
    cov = np.einsum('nti,ntj->nij', center, center) / (r.shape[1] - 1)
    diag = np.diagonal(cov, axis1=1, axis2=2)
    std = np.sqrt(diag)
    corr = cov / (std[:, :, None] * std[:, None, :])
    mask = ~np.eye(r.shape[2], dtype=bool)
    gross = np.exp(r).mean(axis=2)
    logp = np.log(gross)
    common = r.mean(axis=2).var(axis=1, ddof=1)
    idio = (r - r.mean(axis=2, keepdims=True)).var(axis=1, ddof=1).mean(axis=1)
    assert np.allclose(diag.mean(axis=1), common + idio)
    return pd.DataFrame({
        'return': np.prod(gross, axis=1) - 1,
        'volatility': logp.std(axis=1, ddof=1) * np.sqrt(252),
        'corr': corr[:, mask].mean(axis=1),
        'mean_asset_variance': diag.mean(axis=1),
        'mean_offdiag_covariance': cov[:, mask].mean(axis=1),
        'common_variance': common,
        'idiosyncratic_variance': idio,
        'mean_asset_volatility': std.mean(axis=1) * np.sqrt(252),
        'high_invalid': (raw[..., 1::3] < 0).mean(axis=(1, 2)),
        'low_invalid': (raw[..., 2::3] > 0).mean(axis=(1, 2)),
    }), corr


def summarize(raw, targets, cutoff):
    s, _ = stats(raw)
    e = s[['return', 'volatility']].to_numpy() - targets
    return {**{k: float(s[k].mean()) for k in s},
            'n_paths': len(raw), 'target_tail': float((targets[:, 0] <= cutoff).mean()),
            'actual_tail': float((s['return'] <= cutoff).mean()),
            'return_mae': float(np.abs(e[:, 0]).mean()), 'return_bias': float(e[:, 0].mean()),
            'volatility_mae': float(np.abs(e[:, 1]).mean()), 'volatility_bias': float(e[:, 1].mean())}


def main():
    manifest = json.loads((BATCH / 'generation.json').read_text())
    for f, h in manifest['artifacts_sha256'].items():
        verify_historical_hash(BATCH / f, h)
    for f, h in manifest['source_sha256'].items():
        verify_historical_hash(ROOT / f, h)
    assert sha(manifest['data']) == manifest['data_sha256']
    assert sha(manifest['checkpoint']) == manifest['checkpoint_sha256']
    z = np.load(BATCH / 'paths.npz', allow_pickle=False)
    plan = pd.read_csv(BATCH / 'sampling_plan.csv')
    population = pd.read_csv(BATCH / 'population.csv')
    meta = json.loads((TRAIN / 'metadata.json').read_text())
    _, features = load_features(manifest['data'])
    rawall = features[COLUMNS].to_numpy()
    window = lambda starts: np.stack([rawall[s:s+21] for s in starts])
    pop_raw = window(population.window_id)
    refs = window(plan.window_id)[z['draw_id']]
    raw = z['raw_features'].astype(float)
    np.testing.assert_allclose(refs, z['paired_real_future'][z['draw_id']], atol=1e-8)
    cutoff = float(np.quantile(population.target_return, .3))
    assert abs(cutoff - manifest['sampling']['tail_return_cutoff']) < 1e-14
    target = z['target']
    real_s, real_corr = stats(refs)
    gen_s, gen_corr = stats(raw)
    np.testing.assert_allclose(real_s[['return','volatility']], target, atol=1e-12)
    close = z['close'].astype(float)
    with_anchor = np.concatenate([z['anchor_close'][:, None], close], axis=1)
    from_prices = raw.copy()
    from_prices[..., ::3] = np.log(with_anchor[:, 1:] / with_anchor[:, :-1])
    from_prices[..., 1::3] = np.log(z['high'].astype(float) / close)
    from_prices[..., 2::3] = np.log(z['low'].astype(float) / close)
    rebuilt_s, rebuilt_corr = stats(from_prices)
    after_clip = raw.copy()
    after_clip[..., 1::3] = np.maximum(after_clip[..., 1::3], 0)
    after_clip[..., 2::3] = np.minimum(after_clip[..., 2::3], 0)
    clipped_s, _ = stats(after_clip)
    np.testing.assert_array_equal(clipped_s[['return','volatility','corr']], gen_s[['return','volatility','corr']])
    baselines = []
    def add(label, values, weights=None):
        s, _ = stats(values)
        weights = np.full(len(s), 1/len(s)) if weights is None else np.asarray(weights)
        baselines.append({'source': label, 'n_windows_or_paths': len(s),
                          **{k: float(np.sum(s[k] * weights)) for k in s},
                          'tail_share': float(np.sum((s['return'] <= cutoff) * weights))})
    splits = split_windows(features, meta['target_start'], meta['train_end'], meta['val_end'])
    add('generator_training_2005_2017', window(splits['train']))
    add('condition_pool_2010_2017', pop_raw)
    add('expected_70_30_mixture', pop_raw, population.mixture_probability)
    add('selected_paired_real', refs)
    add('generated_raw', raw)
    add('generated_saved_prices', from_prices)
    pd.DataFrame(baselines).to_csv(OUT/'layer_comparison.csv', index=False)
    ticks = z['tickers'].tolist()
    pd.DataFrame(real_corr.mean(axis=0), index=ticks, columns=ticks).to_csv(OUT/'real_correlation_matrix.csv')
    pd.DataFrame(gen_corr.mean(axis=0), index=ticks, columns=ticks).to_csv(OUT/'generated_correlation_matrix.csv')
    rc, gc = real_corr.mean(axis=0), gen_corr.mean(axis=0)
    pair_rows = [{'asset1':ticks[i], 'asset2':ticks[j], 'real':rc[i,j], 'generated':gc[i,j],
                  'difference':gc[i,j]-rc[i,j]} for i in range(10) for j in range(i+1,10)]
    pd.DataFrame(pair_rows).to_csv(OUT/'pairwise_correlation_gaps.csv', index=False)
    asset_rows, support_rows = [], []
    for j, ticker in enumerate(ticks):
        asset_rows.append({'ticker':ticker,
                          'real_volatility':refs[...,3*j].std(axis=1,ddof=1).mean()*np.sqrt(252),
                          'generated_volatility':raw[...,3*j].std(axis=1,ddof=1).mean()*np.sqrt(252)})
        for offset, name in [(1,'high'),(2,'low')]:
            for source, arr in [('paired_real',refs),('generated_raw',raw),('generated_rebuilt',from_prices)]:
                x = arr[...,3*j+offset].ravel()
                invalid = x < 0 if offset == 1 else x > 0
                support_rows.append({'ticker':ticker,'feature':name,'source':source,'n_asset_days':len(x),
                    'invalid_fraction':invalid.mean(),'exact_zero_fraction':(x==0).mean(),
                    'invalid_above_1e_minus_4':(invalid & (np.abs(x)>1e-4)).mean(),
                    'mean':x.mean(),'std':x.std(),'q01':np.quantile(x,.01),'q50':np.quantile(x,.5),
                    'q99':np.quantile(x,.99),
                    'invalid_abs_median':np.median(np.abs(x[invalid])) if invalid.any() else 0})
    pd.DataFrame(asset_rows).to_csv(OUT/'asset_volatility.csv', index=False)
    pd.DataFrame(support_rows).to_csv(OUT/'high_low_support.csv', index=False)
    ids = z['draw_id']
    path_plan = plan.iloc[ids].reset_index(drop=True)
    cutpoints = np.quantile(population.target_return,[.1,.2,.3,.4,.5])
    buckets = pd.cut(target[:,0],[-np.inf,*cutpoints,np.inf],labels=['PR0–10','PR10–20','PR20–30','PR30–40','PR40–50','PR50–100'])
    actual = gen_s['return'].to_numpy()
    tail = target[:,0] <= cutoff
    actual_tail = actual <= cutoff
    transitions = []
    for t in (True,False):
        for g in (True,False):
            mask=(tail==t)&(actual_tail==g)
            transitions.append({'target_tail':t,'generated_tail':g,'n_paths':int(mask.sum())})
    pd.DataFrame(transitions).to_csv(OUT/'tail_transitions.csv',index=False)
    cohorts = []
    for dimension, labels in [('source_pool',path_plan.source_pool),('target_return_pr',buckets),
                               ('target_year',path_plan.target_start.str[:4])]:
        for label in pd.unique(labels):
            mask=np.asarray(labels==label)
            s=summarize(raw[mask],target[mask],cutoff)
            s.update({'dimension':dimension,'group':str(label),
                      'paired_real_corr':float(real_s.loc[mask,'corr'].mean()),
                      'tail_to_non_tail':int((mask&tail&~actual_tail).sum()),
                      'non_tail_to_tail':int((mask&~tail&actual_tail).sum())})
            cohorts.append(s)
    pd.DataFrame(cohorts).to_csv(OUT/'cohort_diagnostics.csv',index=False)
    e=actual-target[:,0]
    loss=tail&~actual_tail
    gain=~tail&actual_tail
    logs=pd.read_csv(TRAIN/'training.csv')
    population_corr=baselines[1]['corr']
    expected_corr=baselines[2]['corr']
    selected_corr=baselines[3]['corr']
    generated_corr=baselines[4]['corr']
    summary={
       'status':'complete','n_paths':len(raw),'data_sha256':manifest['data_sha256'],
       'checkpoint_sha256':manifest['checkpoint_sha256'],'old_batch_manifest_sha256':sha(BATCH/'generation.json'),
       'reconstruction':{'max_abs_return_change':float(np.max(np.abs(rebuilt_s['return']-gen_s['return']))),
                         'max_abs_volatility_change':float(np.max(np.abs(rebuilt_s.volatility-gen_s.volatility))),
                         'max_abs_pair_correlation_change':float(np.max(np.abs(rebuilt_corr-gen_corr))),
                         'high_low_clip_changes_close_statistics':False,
                         'tail_membership_changes':int(((rebuilt_s['return']<=cutoff)!=actual_tail).sum())},
       'correlation_decomposition':{'population':population_corr,'expected_mixture':expected_corr,
             'selected_real':selected_corr,'generated':generated_corr,
             'mix_effect':expected_corr-population_corr,'finite_draw_effect':selected_corr-expected_corr,
             'generation_gap':generated_corr-selected_corr,
             'n_asset_pairs_with_negative_gap':int(sum(r['difference']<0 for r in pair_rows)),
             'idiosyncratic_variance_ratio':baselines[4]['idiosyncratic_variance']/baselines[3]['idiosyncratic_variance'],
             'common_variance_ratio':baselines[4]['common_variance']/baselines[3]['common_variance'],
             'offdiag_covariance_ratio':baselines[4]['mean_offdiag_covariance']/baselines[3]['mean_offdiag_covariance']},
       'tail':{'cutoff':cutoff,'target_count':int(tail.sum()),'actual_count':int(actual_tail.sum()),
             'lost_tail':int(loss.sum()),'entered_tail':int(gain.sum()),
             'net_loss':int(loss.sum()-gain.sum()),'lost_tail_target_within_1pp_cutoff':int((loss&(cutoff-target[:,0]<=.01)).sum()),
             'lost_tail_target_within_2pp_cutoff':int((loss&(cutoff-target[:,0]<=.02)).sum()),
             'return_error_mean':float(e.mean()),'return_error_std':float(e.std()),
             'tail_share_after_removing_global_mean_error_descriptive_only':float((actual-e.mean()<=cutoff).mean()),
             'within_draw_return_std_mean':float(pd.DataFrame({'draw_id':ids,'return':actual}).groupby('draw_id')['return'].std().mean())},
       'loss_log':{'n_logged_steps':len(logs),'median_weighted_condition_over_diffusion':float((logs.weighted_condition_loss/logs.diffusion_loss).median()),
                   'best_step':meta['best_step'],'last_step':int(logs.step.iloc[-1])},
       'validation_selection':{'n_windows':len(splits['validation']),'n_scored':128,
             'first_scored_target':str(features.index[splits['validation'][0]].date()),
             'last_scored_target_end':str(features.index[splits['validation'][127]+20].date()),
             'all_last_target_end':str(features.index[splits['validation'][-1]+20].date())},
       'summary':summarize(raw,target,cutoff),
       'limits':['Covariance decomposition is descriptive, not a trained-loss ablation.',
                 'No PPO, no new training and no production changes.',
                 'Overlapping historical windows are dependent; no IID confidence intervals are claimed.']}
    (OUT/'baseline_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()

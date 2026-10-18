"""Decompose the frozen batch's replicate covariance and support violations."""
import json
import numpy as np
import pandas as pd
from diagnose import OUT, BATCH, stats

z=np.load(BATCH/'paths.npz',allow_pickle=False)
x=z['raw_features'].astype(float).reshape(500,4,21,30)
mean=x.mean(axis=1)
residual=x-mean[:,None]
total,_=stats(x.reshape(2000,21,30))
center,_=stats(mean)
noise,_=stats(residual.reshape(2000,21,30))
rows=[]
for name,frame in [('full_paths',total),('four_replica_mean',center),('replica_residual',noise)]:
    rows.append({'component':name,**{k:float(frame[k].mean()) for k in
                 ['corr','mean_asset_variance','mean_offdiag_covariance','common_variance','idiosyncratic_variance']}})
for key in ['mean_asset_variance','mean_offdiag_covariance','common_variance','idiosyncratic_variance']:
    assert np.isclose(rows[0][key], rows[1][key]+rows[2][key],atol=1e-15)
pd.DataFrame(rows).to_csv(OUT/'replicate_covariance_decomposition.csv',index=False)
support=[]
for name,offset in [('high',1),('low',2)]:
    a=x[...,offset::3]
    m=mean[...,offset::3]
    invalid=a<0 if name=='high' else a>0
    mean_valid=m>=0 if name=='high' else m<=0
    magnitude=np.abs(a[invalid])
    ref=z['paired_real_future'][...,offset::3].astype(float)
    ref_bad=ref<0 if name=='high' else ref>0
    support.append({'feature':name,'n_asset_days':int(a.size),'n_invalid':int(invalid.sum()),
        'invalid_fraction':float(invalid.mean()),
        'fraction_invalid_with_valid_four_replica_mean':float((invalid & mean_valid[:,None]).sum()/invalid.sum()),
        'invalid_abs_median':float(np.median(magnitude)),
        'invalid_abs_q90':float(np.quantile(magnitude,.9)),
        'invalid_abs_q99':float(np.quantile(magnitude,.99)),
        'fraction_invalid_abs_le_1e_minus_6':float((magnitude<=1e-6).mean()),
        'paired_real_invalid_abs_max':float(np.abs(ref[ref_bad]).max()) if ref_bad.any() else 0,
        'paired_real_zero_fraction':float((ref==0).mean()),
        'generated_final_zero_fraction':float((z[name]==z['close']).mean())})
summary={'replicate_covariance':rows,'support':support,
         'limits':['Four-replica averages are finite Monte Carlo estimates, not the exact conditional expectation.',
                   'The residual decomposition is algebraic and does not identify which training change would cure it.']}
(OUT/'noise_and_support.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))

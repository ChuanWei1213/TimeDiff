"""Inspect clean-data predictions along one original 20-step reverse chain."""
import json
import numpy as np
import pandas as pd
import torch
from diagnose import OUT, BATCH, TRAIN, summarize
from data_provider.dow10_cond import load_features
from data_provider.joint_conditions import paired_inputs
from sample_joint_dow10 import load_conditioned_model, generate_raw
from models_diffusion.samplers.dpm_solver import NoiseScheduleVP

torch.set_num_threads(4)
m=json.loads((BATCH/'generation.json').read_text())
model,meta,fs,cs=load_conditioned_model(TRAIN/'model.pt',device='cpu',train_start='2010-01-01',
                                     train_end='2017-12-31',data_sha256=m['data_sha256'])
_,f=load_features(m['data'])
plan=pd.read_csv(BATCH/'sampling_plan.csv').iloc[:8]
h,_,c=paired_inputs(f,plan)
targets=np.repeat(c,4,axis=0)
ns=NoiseScheduleVP('discrete',alphas_cumprod=model.diffusion_worker.alphas_cumprod)
original=model.sampler.model.forward
rows=[]
def tracked(x,t,cond):
    v=original(x,t,cond)
    continuous=(t+1)/ns.total_N
    alpha=ns.marginal_alpha(continuous).reshape(-1,1,1)
    sigma=ns.marginal_std(continuous).reshape(-1,1,1)
    x0=alpha*x-sigma*v
    raw=fs.inverse(x0.permute(0,2,1).cpu().numpy())
    rows.append({'phase':'denoiser_x0','call':len(rows),'model_time':float(t[0]),
                 **summarize(raw,targets,m['sampling']['tail_return_cutoff'])})
    return v
model.sampler.model.forward=tracked
try:
    raw,_=generate_raw(model,h,c,fs,cs,paths_per_condition=4,batch_size=32,diffusion_seed=20261018)
finally:
    model.sampler.model.forward=original
rows.append({'phase':'dpm_return','call':len(rows),'model_time':0.,
             **summarize(raw,targets,m['sampling']['tail_return_cutoff'])})
old=np.load(BATCH/'paths.npz',allow_pickle=False)['raw_features'][:32]
np.testing.assert_array_equal(raw,old)
pd.DataFrame(rows).to_csv(OUT/'reverse_chain_trace.csv',index=False)
print(pd.DataFrame(rows)[['phase','call','model_time','corr','high_invalid','low_invalid','return_mae','volatility_mae']].to_string(index=False))

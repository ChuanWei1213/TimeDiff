"""Replay the original DataLoader RNG path, then extend to all validation windows."""
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from diagnose import OUT, ROOT, BATCH, TRAIN, stats, summarize
from data_provider.dow10_cond import load_features, split_windows, Dow10Windows, COLUMNS, window_conditions
from sample_joint_dow10 import load_conditioned_model
from dow10_cond_experiment import decoder

torch.set_num_threads(4)
m=json.loads((BATCH/'generation.json').read_text())
model,meta,fs,cs=load_conditioned_model(TRAIN/'model.pt',device='cpu',train_start='2010-01-01',
                                     train_end='2017-12-31',data_sha256=m['data_sha256'])
_,f=load_features(m['data'])
starts=split_windows(f,meta['target_start'],meta['train_end'],meta['val_end'])['validation']
dataset=Dow10Windows(f,starts,fs,cs)
loader=DataLoader(dataset,batch_size=32,shuffle=False)
values=[]
torch.manual_seed(1729)
# Constructing the iterator consumes a base-seed draw even with shuffle=False.
# Use the actual loader so the first 128 paths exactly replay validation_score().
with torch.no_grad():
    for history,_,condition in loader:
        pred=model(history,None,decoder(history),None,sample_times=1,market_condition=condition)[0][:,0]
        values.append(fs.inverse(pred.numpy()))
raw=np.concatenate(values)
targets=window_conditions(f[COLUMNS].to_numpy(),starts)
measured,_=stats(raw)
errors=cs.transform(measured[['return','volatility']].to_numpy())-dataset.conditions
mse=np.square(errors).mean(axis=1)
assert abs(float(mse[:128].mean())-meta['best_validation_condition_mse'])<1e-7
rows=[]
for name,idx in [('original_first_128',np.arange(128)),('remaining_355',np.arange(128,len(raw))),('all_483',np.arange(len(raw)))]:
    rows.append({'group':name,**summarize(raw[idx],targets[idx],m['sampling']['tail_return_cutoff']),
                 'standardized_condition_mse':float(mse[idx].mean()),
                 'first_target':str(f.index[starts[idx[0]]].date()),
                 'last_target_end':str(f.index[starts[idx[-1]]+20].date())})
pd.DataFrame(rows).to_csv(OUT/'validation_exact_comparison.csv',index=False)
np.savez_compressed(OUT/'validation_exact.npz',raw_features=raw,targets=targets,window_ids=starts)
(OUT/'validation_exact_audit.json').write_text(json.dumps({
 'status':'passed','recorded_mse':meta['best_validation_condition_mse'],
 'replayed_first128_mse':float(mse[:128].mean()),'total_windows':len(raw),
 'definition':'Original DataLoader and seed 1729; first 128 reproduce checkpoint selection, extended unchanged to all 483.',
 'note':'The manual-tensor batching run in probes/validation_comparison.csv uses a different RNG stream and is not used to reproduce the recorded score.'},indent=2)+'\n')
print(pd.DataFrame(rows).to_string(index=False))

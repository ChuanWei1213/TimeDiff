"""Frozen-checkpoint inference probes. No training and no original artifact writes."""
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import torch

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
sys.path.insert(0,str(ROOT))
from diagnose import stats, summarize, sha, BATCH, TRAIN
from data_provider.dow10_cond import load_features, COLUMNS, split_windows, window_conditions
from data_provider.joint_conditions import paired_inputs
from sample_joint_dow10 import load_conditioned_model, generate_raw
from dow10_cond_experiment import decoder


def main():
    dest=OUT/'probes'
    dest.mkdir(exist_ok=False)
    torch.set_num_threads(4)
    manifest=json.loads((BATCH/'generation.json').read_text())
    model,meta,fs,cs=load_conditioned_model(TRAIN/'model.pt',device='cpu',train_start='2010-01-01',
                       train_end='2017-12-31',data_sha256=manifest['data_sha256'])
    _,features=load_features(manifest['data'])
    # Prefix of the already randomized plan, selected before examining probe results.
    plan=pd.read_csv(BATCH/'sampling_plan.csv').iloc[:96].copy()
    h,real,c=paired_inputs(features,plan)
    cutoff=manifest['sampling']['tail_return_cutoff']
    ids=np.repeat(np.arange(len(plan)),4)
    rows=[]
    for seed,steps in [(20261018,20),(20261018,50),(20261018,100),(20261019,20),(20261019,100)]:
        model.args.sampling_steps=steps
        start=time.monotonic()
        raw,_=generate_raw(model,h,c,fs,cs,paths_per_condition=4,batch_size=32,diffusion_seed=seed)
        row={'seed':seed,'sampling_steps':steps,'seconds':time.monotonic()-start,
             **summarize(raw,c[ids],cutoff)}
        if seed==20261018 and steps==20:
            old=np.load(BATCH/'paths.npz',allow_pickle=False)['raw_features'][:len(raw)]
            row['max_abs_replay_error']=float(np.max(np.abs(raw-old)))
            np.testing.assert_array_equal(raw,old)
        np.savez_compressed(dest/f'seed_{seed}_steps_{steps}.npz',raw_features=raw,targets=c[ids],draw_ids=ids)
        rows.append(row)
        pd.DataFrame(rows).to_csv(dest/'solver_comparison.csv',index=False)
        print(json.dumps(row),flush=True)
    # Eval-mode single denoising steps: deliberately corrupted real futures.
    # They are diagnostic targets, never production inference conditioning.
    model.args.sampling_steps=20
    ht=torch.as_tensor(fs.transform(h))
    ct=torch.as_tensor(cs.transform(c))
    xt=torch.as_tensor(fs.transform(real)).permute(0,2,1)
    with torch.inference_mode():
        ar=model.dlinear_model(ht.permute(0,2,1)[:,:,-model.short_term_range:])
        cond={'history':torch.cat([ht.permute(0,2,1),ar],dim=-1),'market':ct}
        assert not model.args.use_future_mixup and not model.args.use_window_normalization
        teacher=[]
        for step,pure in [(0,False),(25,False),(50,False),(75,False),(99,False),(99,True)]:
            torch.manual_seed(20261020)
            noise=torch.randn_like(xt)
            t=torch.full((len(ht),),step,dtype=torch.long)
            noisy=noise if pure else model.diffusion_worker.q_sample(xt,t,noise)
            pred=model.diffusion_worker.nn(noisy,t,cond,y_clean=None)
            x0=model.diffusion_worker._to_x0(pred,noisy,t)
            result=fs.inverse(x0.permute(0,2,1).numpy())
            row={'timestep':step,'input':'pure_noise' if pure else 'noised_real_future',
                 'sqrt_alpha_bar':float(model.diffusion_worker.sqrt_alphas_cumprod[step]),
                 **summarize(result,c,cutoff)}
            teacher.append(row)
        pd.DataFrame(teacher).to_csv(dest/'denoising_comparison.csv',index=False)
        # Full generator validation period, one path/window and original validation RNG.
        starts=split_windows(features,meta['target_start'],meta['train_end'],meta['val_end'])['validation']
        data=features[COLUMNS].to_numpy()
        vh=np.stack([data[s-63:s] for s in starts])
        vc=window_conditions(data,starts)
        vr=[]
        torch.manual_seed(1729)
        for b in range(0,len(vh),32):
            history=torch.as_tensor(fs.transform(vh[b:b+32]))
            condition=torch.as_tensor(cs.transform(vc[b:b+32]))
            result=model(history,None,decoder(history),None,sample_times=1,market_condition=condition)[0][:,0]
            vr.append(fs.inverse(result.numpy()))
        vr=np.concatenate(vr)
        np.savez_compressed(dest/'full_validation.npz',raw_features=vr,targets=vc,window_ids=starts)
        validation=[]
        vm,_=stats(vr)
        vz=cs.transform(vm[['return','volatility']].to_numpy())-cs.transform(vc)
        for label,sel in [('original_first_128',np.arange(128)),('remaining_355',np.arange(128,len(vh))),
                          ('all_483',np.arange(len(vh)))]:
            row={'group':label,**summarize(vr[sel],vc[sel],cutoff),
                 'standardized_condition_mse':float(np.square(vz[sel]).mean()),
                 'first_target':str(features.index[starts[sel[0]]].date()),
                 'last_target_end':str(features.index[starts[sel[-1]]+20].date())}
            validation.append(row)
        pd.DataFrame(validation).to_csv(dest/'validation_comparison.csv',index=False)
    result={'status':'complete','n_draws':len(plan),'n_unique_windows':int(plan.window_id.nunique()),
            'subset':'first 96 draws of the previously shuffled frozen 500-draw plan',
            'n_paths_per_solver_run':len(ids),'paths_per_condition':4,
            'model_sha256':sha(TRAIN/'model.pt'),'data_sha256':sha(manifest['data']),
            'validation_mse_first128':validation[0]['standardized_condition_mse'],
            'checkpoint_recorded_validation_mse':meta['best_validation_condition_mse'],
            'training_performed':False,'ppo_performed':False,
            'caveats':['Solver comparison is a paired inference intervention on a small fixed subset, not a new production batch.',
                       'Noised-real-future denoising has access to true-future signal via x_t and is not a generation evaluation.',
                       'Full validation audit is of the selected checkpoint only; other training checkpoints were not saved.']}
    assert result['model_sha256']==manifest['checkpoint_sha256']
    assert result['data_sha256']==manifest['data_sha256']
    (dest/'manifest.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    main()

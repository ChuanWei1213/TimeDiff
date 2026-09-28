"""Run the lambda=0.1 SPX experiment and record actual wall-clock phase times."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tag', default='condition_lambda_0p1')
    cli = parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_-]+', cli.tag):
        raise ValueError('Use letters, digits, hyphens or underscores in --tag')
    baseline = ROOT/'outputs/spx/crash_20260929'
    for path in [ROOT/'datasets/spx/prices.csv', baseline/'no_mixup_samples/paths.csv',
                 baseline/'train/metadata.json', ROOT/'notebooks/spx_condition_loss_comparison.ipynb']:
        if not path.is_file():
            raise FileNotFoundError(path)
    run = ROOT/'outputs/spx'/cli.tag
    run.mkdir()  # Atomic; never overwrite a previous experiment.
    env = dict(os.environ, SPX_CONSISTENCY_RUN=str(run), PYTHONUNBUFFERED='1')
    env.setdefault('MPLCONFIGDIR', str(run/'matplotlib'))
    env.setdefault('IPYTHONDIR', str(run/'ipython'))
    conditions = [(-.05, .15), (.05, .15), (-.05, .35), (.05, .35),
                  (-.10, .40), (-.20, .60), (-.30, .80)]
    commands = [
        ('training', [sys.executable, '-u', 'spx_experiment.py', 'train',
                      '--train-end', '2021-12-31', '--val-end', '2023-12-31',
                      '--ar-steps', '500', '--steps', '2000', '--eval-every', '100', '--val-batches', '4',
                      '--batch-size', '32', '--seed', '2024', '--device', 'cpu',
                      '--lambda-condition', '0.1', '--out', str(run/'train')]),
        ('sampling', [sys.executable, '-u', 'spx_experiment.py', 'sample',
                      '--checkpoint', str(run/'train/model.pt'), '--history-end', '2023-12-29',
                      *[s for r, v in conditions for s in ['--condition', str(r), str(v)]],
                      '--n-paths', '200', '--batch-size', '32', '--seed', '2024', '--device', 'cpu',
                      '--out', str(run/'samples')]),
        ('notebook', [sys.executable, '-u', 'notebooks/execute_in_process.py',
                      'notebooks/spx_condition_loss_comparison.ipynb']),
    ]
    started = perf_counter()
    record = {'started_at_utc': datetime.now(timezone.utc).isoformat(), 'lambda_condition': .1,
              'baseline': str(baseline), 'baseline_reused': True, 'stages': [],
              'note': 'Wall-clock seconds; process durations include imports. Training subphases are in train/timings.json.'}
    timing_path = run/'pipeline_timings.json'
    for name, command in commands:
        print(f'\nStage {name}: {" ".join(command)}', flush=True)
        row = {'name': name, 'command': command, 'started_at_utc': datetime.now(timezone.utc).isoformat(), 'status': 'running'}
        record['stages'].append(row)
        timing_path.write_text(json.dumps(record, indent=2)+'\n')
        stage_started = perf_counter()
        with (run/f'{name}.log').open('wb') as log:
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            while chunk := process.stdout.read1(65536):
                log.write(chunk)
                log.flush()
                sys.stdout.buffer.write(chunk)
                sys.stdout.buffer.flush()
            code = process.wait()
        row.update(seconds=perf_counter()-stage_started, returncode=code, status='complete' if code == 0 else 'failed')
        record['elapsed_seconds'] = perf_counter()-started
        timing_path.write_text(json.dumps(record, indent=2)+'\n')
        if code:
            raise SystemExit(code)
    record['status'] = 'complete'
    timing_path.write_text(json.dumps(record, indent=2)+'\n')
    print(f'\nComplete; timings: {timing_path}', flush=True)


if __name__ == '__main__':
    main()

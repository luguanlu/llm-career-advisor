#!/usr/bin/env python3
"""Run evaluation for each saved bohb stage1 model directory.
Creates a temp YAML per stage with do_train: false and do_eval: true, and runs
`python3 -m llamafactory.cli train <temp_yaml>` to trigger evaluation pipeline.
"""
import yaml
from pathlib import Path
import subprocess
import shutil

BASE = Path(__file__).resolve().parents[2]
EXPS = BASE / 'experiments' / 'bohb'
RESULTS = BASE / 'saves' / 'bohb' / 'llama3-8b' / 'stage1'

stages = sorted([p.name for p in RESULTS.iterdir() if p.name.startswith('bohb_')])

for stage in stages:
    yaml_src = EXPS / f'bohb_stage1_{stage.split("_")[-1]}.yaml'
    if not yaml_src.exists():
        print('skip', stage, 'no yaml')
        continue
    with open(yaml_src, 'r', encoding='utf-8') as f:
        conf = yaml.safe_load(f)

    # override for eval
    conf['do_train'] = False
    conf['do_eval'] = True
    # point model to the saved model dir so loader loads the checkpoint
    conf['model_name_or_path'] = str(RESULTS / stage)
    # set output_dir to existing stage dir
    conf['output_dir'] = str(RESULTS / stage)
    # ensure small eval batch
    conf['per_device_eval_batch_size'] = conf.get('per_device_eval_batch_size', 1)

    temp = EXPS / f'{yaml_src.stem}_eval.yaml'
    with open(temp, 'w', encoding='utf-8') as f:
        yaml.safe_dump(conf, f, sort_keys=False)

    print('Evaluating', stage)
    cmd = ['python3', '-m', 'llamafactory.cli', 'train', str(temp)]
    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        print('Eval failed for', stage)
    # remove temp yaml
    try:
        temp.unlink()
    except Exception:
        pass

print('Done')

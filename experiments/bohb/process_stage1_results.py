#!/usr/bin/env python3
import os, re, yaml, csv
from pathlib import Path
import math

root = Path(__file__).resolve().parents[2]
cfg_dir = root / 'experiments' / 'bohb'
logs_dir = cfg_dir / 'logs'
fig_dir = root / 'saves' / 'bohb' / 'llama3-8b' / 'stage1' / 'figures'
fig_dir.mkdir(parents=True, exist_ok=True)

def extract_losses_from_log(logtext):
    # common patterns
    train_loss = None
    eval_loss = None
    # regex: look for 'train_loss: 1.234' or 'loss: 1.234' lines with 'eval' nearby
    for line in logtext.splitlines():
        l = line.lower()
        m = re.search(r"eval[_ ]?loss[:=]\s*([0-9]+\.?[0-9eE+-]*)", l)
        if m:
            try:
                eval_loss = float(m.group(1))
            except:
                pass
        m2 = re.search(r"train[_ ]?loss[:=]\s*([0-9]+\.?[0-9eE+-]*)", l)
        if m2:
            try:
                train_loss = float(m2.group(1))
            except:
                pass
        # fallback: 'loss: 1.234' (may be train)
        if train_loss is None:
            m3 = re.search(r"\bloss[:=]\s*([0-9]+\.?[0-9eE+-]*)", l)
            if m3:
                try:
                    train_loss = float(m3.group(1))
                except:
                    pass
    return train_loss, eval_loss

rows = []
files = sorted(cfg_dir.glob('bohb_stage1_*.yaml'))
for cfgfile in files:
    name = cfgfile.stem
    # load params
    with open(cfgfile,'r') as fh:
        cfg = yaml.safe_load(fh)
    params = {
        'config': name,
        'lora_rank': cfg.get('lora_rank'),
        'lora_alpha': cfg.get('lora_alpha'),
        'lora_dropout': cfg.get('lora_dropout'),
        'learning_rate': cfg.get('learning_rate'),
        'warmup_ratio': cfg.get('warmup_ratio'),
        'lr_scheduler_type': cfg.get('lr_scheduler_type'),
        'output_dir': cfg.get('output_dir')
    }
    logfile = logs_dir / f"{name}.log"
    error = ''
    train_loss = None
    eval_loss = None
    completed = False
    if logfile.exists():
        text = logfile.read_text(errors='ignore')
        if 'no such file or directory' in text.lower() or 'error' in text.lower():
            # capture first error line
            lines = [l for l in text.splitlines() if l.strip()]
            error = '\\n'.join(lines[:5])
        t,e = extract_losses_from_log(text)
        train_loss = t
        eval_loss = e
        # heuristics for completion
        if re.search(r'training (completed|finished)|saved checkpoint|saving model|finished training', text, re.I):
            completed = True
    else:
        error = 'log missing'

    rows.append({**params,'train_loss': train_loss, 'eval_loss': eval_loss, 'completed': completed, 'error': error})

# write CSV
out_csv = cfg_dir / 'results_summary.csv'
with open(out_csv, 'w', newline='') as csvf:
    writer = csv.DictWriter(csvf, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    for r in rows:
        writer.writerow({k: ('' if v is None else v) for k,v in r.items()})

print('Wrote', out_csv)

# plotting
import matplotlib.pyplot as plt
import numpy as np

def safe_list(vals):
    return [v for v in vals if v is not None]

names = [r['config'] for r in rows]
evals = [r['eval_loss'] for r in rows]
trains = [r['train_loss'] for r in rows]
ranks = [r['lora_rank'] for r in rows]
lrs = [r['learning_rate'] for r in rows]
drops = [r['lora_dropout'] for r in rows]

fig_dir = Path(fig_dir)

def plot_eval_bar():
    if all(v is None for v in evals):
        plt.figure(figsize=(8,4))
        plt.text(0.5,0.5,'No eval_loss available in logs',ha='center',va='center')
        plt.axis('off')
        plt.savefig(fig_dir / 'eval_loss_bar.png')
        plt.close()
        return
    x = np.arange(len(names))
    y = [v if v is not None else np.nan for v in evals]
    plt.figure(figsize=(12,6))
    plt.bar(x,y)
    plt.xticks(x,names,rotation=45,ha='right')
    plt.ylabel('eval_loss')
    plt.tight_layout()
    plt.savefig(fig_dir / 'eval_loss_bar.png')
    plt.close()

def plot_train_eval():
    if all(v is None for v in evals) and all(v is None for v in trains):
        plt.figure(figsize=(8,4))
        plt.text(0.5,0.5,'No train/eval loss available',ha='center',va='center')
        plt.axis('off')
        plt.savefig(fig_dir / 'train_eval_loss_comparison.png')
        plt.close()
        return
    x = np.arange(len(names))
    y_tr = [v if v is not None else np.nan for v in trains]
    y_ev = [v if v is not None else np.nan for v in evals]
    plt.figure(figsize=(12,6))
    plt.plot(x,y_tr,label='train_loss',marker='o')
    plt.plot(x,y_ev,label='eval_loss',marker='x')
    plt.xticks(x,names,rotation=45,ha='right')
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / 'train_eval_loss_comparison.png')
    plt.close()

def scatter_and_save(xvals,yvals,xlabel,filename):
    if all(v is None for v in yvals):
        plt.figure(figsize=(6,4))
        plt.text(0.5,0.5,'No eval_loss available',ha='center',va='center')
        plt.axis('off')
        plt.savefig(fig_dir / filename)
        plt.close()
        return
    x = [xi for xi,yi in zip(xvals,yvals) if yi is not None]
    y = [yi for yi in yvals if yi is not None]
    plt.figure(figsize=(6,4))
    plt.scatter(x,y)
    plt.xlabel(xlabel)
    plt.ylabel('eval_loss')
    plt.tight_layout()
    plt.savefig(fig_dir / filename)
    plt.close()

plot_eval_bar()
plot_train_eval()
scatter_and_save(ranks, evals, 'lora_rank', 'rank_vs_eval_loss.png')
scatter_and_save(lrs, evals, 'learning_rate', 'lr_vs_eval_loss.png')
scatter_and_save(drops, evals, 'lora_dropout', 'dropout_vs_eval_loss.png')

print('Wrote figures to', fig_dir)

import argparse
import csv
import json
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import random
import time
from pathlib import Path
from layout import BUNDLE, CHECKPOINTS, RESULTS, HISTORIES, PREDICTIONS, FIGURES, VECTOR, DEMO, DATA
import numpy as np
import torch
from torch import nn
from dataset import prepare, normalize, augment_batch
from models import make_model

VARIANTS = {'A': ('cnn_baseline', False, False), 'B': ('cnn_aug', False, True),
            'C': ('cnn_se', True, False), 'D': ('cnn_se_aug', True, True)}


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(8)


def sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize()


@torch.inference_mode()
def validate(model, data, device, batch_size=512):
    model.eval(); loss = torch.zeros((), device=device); correct = torch.zeros((), device=device)
    x, y = data['val_x'], data['val_y']
    for start in range(0, len(y), batch_size):
        xb, yb = x[start:start+batch_size], y[start:start+batch_size]
        out = model(normalize(xb, data['mean'], data['std']))
        loss += nn.functional.cross_entropy(out, yb, reduction='sum')
        correct += (out.argmax(1) == yb).sum()
    return loss.item() / len(y), correct.item() / len(y)


def train_one(variant, seed, config, data, root, device):
    name, se, aug = VARIANTS[variant]
    ckpt_dir = CHECKPOINTS / f'seed_{seed}'; ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir = HISTORIES; log_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = ckpt_dir / f'{name}.pth'
    history_path = log_dir / f'{name}_seed{seed}.json'
    timing_path = log_dir / f'{name}_seed{seed}_timing.json'
    if checkpoint.exists() and history_path.exists() and timing_path.exists():
        old = json.loads(history_path.read_text(encoding='utf-8'))
        if len(old) == config['epochs']:
            cached = torch.load(checkpoint, map_location='cpu', weights_only=True)
            if cached['config'] != config:
                raise ValueError('缓存实验配置不同。请使用新目录保存新实验，保留原结果。')
            print(f'已完成，保留 {variant} seed={seed}', flush=True); return
    set_seed(seed)
    model = make_model(se, seed, config['reduction']).to(device)
    # 模型构造之后重新固定 dropout 随机流，使模块初始化消耗不影响训练。
    torch.manual_seed(seed + 200000)
    order_rng = torch.Generator(device=device).manual_seed(seed + 300000)
    aug_rng = torch.Generator(device=device).manual_seed(seed + 400000)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config['epochs'], eta_min=1e-5)
    criterion = nn.CrossEntropyLoss()
    x, y = data['train_x'], data['train_y']; n = len(y)
    history, best_acc, best_loss = [], -1, float('inf')
    sync(device); started = time.perf_counter()
    for epoch in range(1, config['epochs'] + 1):
        epoch_start = time.perf_counter(); model.train()
        total_loss = torch.zeros((), device=device); correct = torch.zeros((), device=device)
        permutation = torch.randperm(n, generator=order_rng, device=device)
        lr = optimizer.param_groups[0]['lr']
        for start in range(0, n, config['batch_size']):
            idx = permutation[start:start+config['batch_size']]
            xb, yb = x[idx], y[idx]
            if aug:
                xb = augment_batch(xb, aug_rng)
            xb = normalize(xb, data['mean'], data['std'])
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb); loss = criterion(logits, yb)
            loss.backward(); optimizer.step()
            total_loss += loss.detach() * len(yb)
            correct += (logits.detach().argmax(1) == yb).sum()
        val_loss, val_acc = validate(model, data, device)
        scheduler.step(); sync(device)
        row = {'epoch': epoch, 'lr': lr, 'train_loss': total_loss.item()/n, 'train_accuracy': correct.item()/n,
               'val_loss': val_loss, 'val_accuracy': val_acc, 'epoch_seconds': time.perf_counter()-epoch_start}
        history.append(row)
        if val_acc > best_acc or (val_acc == best_acc and val_loss < best_loss):
            best_acc, best_loss = val_acc, val_loss
            torch.save({'model_state': model.state_dict(), 'variant': variant, 'name': name, 'seed': seed,
                        'use_se': se, 'augmentation': aug, 'config': config, 'epoch': epoch,
                        'best_val_accuracy': val_acc, 'best_val_loss': val_loss,
                        'normalization': {'mean': data['mean'], 'std': data['std']},
                        'split_seed': config['split_seed']}, checkpoint)
        history_path.write_text(json.dumps(history, indent=2), encoding='utf-8')
        print(f'{variant} seed={seed} [{epoch:02d}/{config["epochs"]}] loss={row["train_loss"]:.4f} train={row["train_accuracy"]:.4%} val={val_acc:.4%} best={best_acc:.4%} {row["epoch_seconds"]:.2f}s', flush=True)
    sync(device); elapsed = time.perf_counter()-started
    stats = {'variant': variant, 'seed': seed, 'training_seconds': elapsed, 'epochs': config['epochs'], 'history_file': str(history_path.relative_to(BUNDLE))}
    (log_dir / f'{name}_seed{seed}_timing.json').write_text(json.dumps(stats, indent=2), encoding='utf-8')
    with history_path.with_suffix('.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=history[0].keys()); writer.writeheader(); writer.writerows(history)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--variant', choices=VARIANTS, default='A'); parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--config', default='configs/experiment.json'); args = parser.parse_args()
    root = Path(__file__).resolve().parent; config = json.loads((root/args.config).read_text(encoding='utf-8'))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = prepare(DATA, RESULTS, device, config['split_seed'])
    train_one(args.variant, args.seed, config, data, root, device)

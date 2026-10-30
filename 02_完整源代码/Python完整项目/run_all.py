"""训练全部模型后才开始最终测试；复现固定配置，不用测试结果调参。"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import csv
import json
import platform
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from layout import BUNDLE, CHECKPOINTS, RESULTS, HISTORIES, PREDICTIONS, FIGURES, VECTOR, DEMO, DATA
import numpy as np
import psutil
import torch
from dataset import prepare
from train import VARIANTS, train_one, set_seed
from test import evaluate_checkpoint
from metrics import paired_comparison


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--stage', choices=['train','evaluate','all'], default='all'); args = parser.parse_args()
    root = Path(__file__).resolve().parent
    cfg = json.loads((root/'configs/experiment.json').read_text(encoding='utf-8'))
    (RESULTS).mkdir(exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_seed(cfg['seeds'][0])
    environment = {'date_local': datetime.now().isoformat(), 'platform': platform.platform(), 'python': sys.version,
                   'python_executable': sys.executable, 'pytorch': torch.__version__, 'cuda_runtime': torch.version.cuda,
                   'cudnn': torch.backends.cudnn.version(), 'cuda_available': torch.cuda.is_available(),
                   'gpu': torch.cuda.get_device_name(0) if device.type=='cuda' else None,
                   'gpu_memory_bytes': torch.cuda.get_device_properties(0).total_memory if device.type=='cuda' else None,
                   'cpu_logical': psutil.cpu_count(), 'cpu_physical': psutil.cpu_count(logical=False),
                   'ram_bytes': psutil.virtual_memory().total, 'deterministic_algorithms': True, 'precision': 'float32',
                   'torch_threads': torch.get_num_threads(), 'config': cfg}
    (RESULTS / 'environment.json').write_text(json.dumps(environment, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(environment, ensure_ascii=False, indent=2), flush=True)
    # RTX 50 系显卡实测执行一个 CUDA 运算以验证 wheel 的架构兼容性。
    print('设备运算检查', (torch.ones(16,16,device=device) @ torch.ones(16,16,device=device)).sum().item(), flush=True)
    data = prepare(DATA, RESULTS, device, cfg['split_seed'])
    if args.stage in ['train','all']:
        for seed in cfg['seeds']:
            for variant in VARIANTS:
                train_one(variant, seed, cfg, data, root, device)
    if args.stage in ['evaluate','all']:
        rows = []
        # 预检保证 12 次训练已完成，避免提前查看部分测试结果。
        for seed in cfg['seeds']:
            for variant,(name,se,aug) in VARIANTS.items():
                assert (CHECKPOINTS/f'seed_{seed}'/f'{name}.pth').exists()
                history = json.loads((HISTORIES/f'{name}_seed{seed}.json').read_text())
                assert len(history) == cfg['epochs']
        for seed in cfg['seeds']:
            for variant,(name,se,aug) in VARIANTS.items():
                rows.append(evaluate_checkpoint(CHECKPOINTS/f'seed_{seed}'/f'{name}.pth', data, root, device))
                if seed == cfg['primary_seed']:
                    shutil.copy2(CHECKPOINTS/f'seed_{seed}'/f'{name}.pth', CHECKPOINTS/f'{name}.pth')
        (RESULTS / 'all_metrics.json').write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding='utf-8')
        scalar_keys = [k for k,v in rows[0].items() if not isinstance(v, list)]
        with (RESULTS / '实验结果汇总.csv').open('w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=scalar_keys, extrasaction='ignore'); writer.writeheader(); writer.writerows(rows)
        summary = []
        for variant,(name,se,aug) in VARIANTS.items():
            subset = [r for r in rows if r['variant']==variant]
            record = {'variant': variant, 'name': name, 'n_seeds': len(subset), 'parameters': subset[0]['parameters'], 'macs_conv_linear':subset[0]['macs_conv_linear']}
            for key in ['accuracy','precision_macro','recall_macro','f1_macro','best_val_accuracy','training_seconds','latency_batch1_ms_median','latency_batch256_ms_median']:
                record[key+'_mean'] = float(np.mean([r[key] for r in subset]))
                record[key+'_std'] = float(np.std([r[key] for r in subset],ddof=1))
            summary.append(record)
        (RESULTS / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
        with (RESULTS / '多种子统计.csv').open('w', newline='', encoding='utf-8-sig') as f:
            writer=csv.DictWriter(f, fieldnames=summary[0].keys()); writer.writeheader(); writer.writerows(summary)
        pred = {v: np.load(PREDICTIONS/f'{name}_seed{cfg["primary_seed"]}.npz') for v,(name,_,_) in VARIANTS.items()}
        comparison = {v: paired_comparison(pred['A']['target'], pred['A']['prediction'], pred[v]['prediction']) for v in ['B','C','D']}
        (RESULTS / 'paired_comparisons.json').write_text(json.dumps(comparison, indent=2), encoding='utf-8')
        print('实验完成，全部指标已保存。', flush=True)


if __name__ == '__main__':
    main()

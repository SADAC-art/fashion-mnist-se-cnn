"""重新载入最佳权重，只在全部训练结束之后调用最终测试。"""
import argparse
import hashlib
import json
import time
from pathlib import Path
from layout import BUNDLE, CHECKPOINTS, RESULTS, HISTORIES, PREDICTIONS, FIGURES, VECTOR, DEMO, DATA
import numpy as np
import torch
from dataset import prepare, normalize, CLASSES
from models import ClothingCNN, count_macs
from metrics import classification_metrics
from train import VARIANTS, sync, set_seed


@torch.inference_mode()
def evaluate_checkpoint(path, data, root, device, benchmark=True):
    ckpt = torch.load(path, map_location='cpu', weights_only=True)
    set_seed(ckpt['seed'])
    model = ClothingCNN(ckpt['use_se'], ckpt['config']['reduction']).to(device)
    model.load_state_dict(ckpt['model_state']); model.eval()
    assert ckpt['normalization'] == {'mean': data['mean'], 'std': data['std']}
    x = normalize(data['test_x'], data['mean'], data['std']); y = data['test_y'].cpu().numpy()
    predictions = torch.cat([model(x[s:s+512]).argmax(1) for s in range(0, len(y), 512)]).cpu().numpy()
    result, report, cm = classification_metrics(y, predictions, CLASSES)
    params = sum(p.numel() for p in model.parameters())
    result.update({'variant': ckpt['variant'], 'name': ckpt['name'], 'seed': ckpt['seed'], 'best_epoch': ckpt['epoch'],
                   'best_val_accuracy': ckpt['best_val_accuracy'], 'parameters': params,
                   'parameter_bytes_fp32': params*4, 'checkpoint_bytes': path.stat().st_size, 'macs_conv_linear': count_macs(model),
                   'checkpoint_sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'device': str(device)})
    timing = HISTORIES/f'{ckpt["name"]}_seed{ckpt["seed"]}_timing.json'
    result['training_seconds'] = json.loads(timing.read_text())['training_seconds']
    if benchmark:
        # 显存驻留数据，模型前向耗时；不包含磁盘读取、预处理或主机传输。
        for batch in [1, 256]:
            inp = x[:batch]
            for _ in range(30): model(inp)
            sync(device); values = []
            for repeat in range(5):
                start = time.perf_counter()
                for _ in range(100): model(inp)
                sync(device); values.append((time.perf_counter()-start)*1000/100)
            result[f'latency_batch{batch}_ms_median'] = float(np.median(values))
            result[f'latency_batch{batch}_ms_all'] = values
    prefix = PREDICTIONS/f'{ckpt["name"]}_seed{ckpt["seed"]}'
    prefix.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(prefix)+'.npz', target=y, prediction=predictions, confusion=cm)
    np.savetxt(str(prefix)+'_confusion.csv', cm, fmt='%d', delimiter=',')
    Path(str(prefix)+'_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    Path(str(prefix)+'_metrics.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'最终测试 {ckpt["variant"]} seed={ckpt["seed"]}: {result["accuracy"]:.4%}; params={params}', flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--checkpoint', required=True); args = parser.parse_args()
    root = Path(__file__).resolve().parent
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cfg = json.loads((root/'configs/experiment.json').read_text())
    data = prepare(DATA, RESULTS, device, cfg['split_seed'])
    evaluate_checkpoint(Path(args.checkpoint), data, root, device)

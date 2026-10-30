"""下载原始 IDX 文件、校验哈希并建立固定分层划分。"""
import gzip
import hashlib
import json
import struct
import urllib.request
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F

CLASSES = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
CLASSES_CN = ['T恤', '裤子', '套衫', '连衣裙', '外套', '凉鞋', '衬衫', '运动鞋', '包', '短靴']
FILES = {
    'train-images-idx3-ubyte.gz': '8d4fb7e6c68d591d4c3dfef9ec88bf0d',
    'train-labels-idx1-ubyte.gz': '25c81989df183df01b3e8a0aad5dffbe',
    't10k-images-idx3-ubyte.gz': 'bef4ecab320f06d8554ea6380940ec79',
    't10k-labels-idx1-ubyte.gz': 'bb300cfdad3c16e7a12a480ee83cd310',
}
BASE_URLS = ['https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/',
             'https://fashion-mnist.s3-website.eu-central-1.amazonaws.com/']


def load_raw(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    arrays, provenance = [], []
    for name, expected in FILES.items():
        path = root / name
        if not path.exists() or hashlib.md5(path.read_bytes()).hexdigest() != expected:
            for base in BASE_URLS:
                try:
                    print('下载', base + name, flush=True)
                    request = urllib.request.Request(base + name, headers={'User-Agent': 'FashionMNIST-course-project'})
                    with urllib.request.urlopen(request, timeout=120) as response:
                        content = response.read()
                    if hashlib.md5(content).hexdigest() != expected:
                        raise ValueError('数据 MD5 不匹配')
                    path.write_bytes(content)
                    break
                except Exception as exc:
                    print(type(exc).__name__, str(exc), flush=True)
            else:
                raise RuntimeError(f'无法下载 {name}')
        blob = gzip.decompress(path.read_bytes())
        magic = struct.unpack('>I', blob[:4])[0]
        if magic == 2051:
            n, h, w = struct.unpack('>III', blob[4:16])
            value = np.frombuffer(blob, dtype=np.uint8, offset=16).reshape(n, h, w).copy()
            assert (h, w) == (28, 28)
        else:
            assert magic == 2049
            n = struct.unpack('>I', blob[4:8])[0]
            value = np.frombuffer(blob, dtype=np.uint8, offset=8).copy()
        assert len(value) == n
        arrays.append(value)
        provenance.append({'file': name, 'md5': expected, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'bytes': path.stat().st_size})
    return (*arrays, provenance)


def prepare(root, result_dir, device, split_seed=314159):
    x, y, xt, yt, provenance = load_raw(root)
    assert x.shape == (60000, 28, 28) and xt.shape == (10000, 28, 28)
    assert np.array_equal(np.bincount(y), np.full(10, 6000))
    assert np.array_equal(np.bincount(yt), np.full(10, 1000))
    rng = np.random.default_rng(split_seed)
    train, val = [], []
    for cls in range(10):
        indices = rng.permutation(np.flatnonzero(y == cls))
        train.extend(indices[:5000]); val.extend(indices[5000:])
    train, val = np.sort(train), np.sort(val)
    assert not np.intersect1d(train, val).size
    assert len(np.union1d(train, val)) == 60000
    mean = float(x[train].mean(dtype=np.float64) / 255)
    std = float(x[train].std(dtype=np.float64) / 255)
    result_dir = Path(result_dir); result_dir.mkdir(parents=True, exist_ok=True)
    np.savez(result_dir / 'split_indices.npz', train=train, val=val)
    audit = {'split_seed': split_seed, 'train_n': len(train), 'val_n': len(val), 'test_n': len(yt),
             'train_per_class': np.bincount(y[train]).tolist(), 'val_per_class': np.bincount(y[val]).tolist(),
             'test_per_class': np.bincount(yt).tolist(), 'mean': mean, 'std': std,
             'normalization_source': 'training subset only', 'files': provenance,
             'split_sha256': hashlib.sha256((result_dir / 'split_indices.npz').read_bytes()).hexdigest()}
    (result_dir / 'dataset_audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding='utf-8')
    def tx(a):
        return torch.from_numpy(a).to(device).float().unsqueeze(1) / 255
    def ty(a):
        return torch.from_numpy(a.astype(np.int64)).to(device)
    # 一次载入显存减少 Windows 多进程加载开销，CPU 同样可用。
    return {'train_x': tx(x[train]), 'train_y': ty(y[train]), 'val_x': tx(x[val]), 'val_y': ty(y[val]),
            'test_x': tx(xt), 'test_y': ty(yt), 'mean': mean, 'std': std, 'audit': audit}


def normalize(x, mean, std):
    return (x - mean) / std


def augment_batch(x, generator):
    # 每张图独立旋转 ±8°、水平/竖直平移 ±2 像素，双线性插值、黑色填充。
    n = x.shape[0]
    r = torch.rand((n, 3), generator=generator, device=x.device)
    angle = (r[:, 0] * 2 - 1) * (8 * np.pi / 180)
    theta = torch.zeros(n, 2, 3, device=x.device)
    theta[:, 0, 0] = theta[:, 1, 1] = angle.cos()
    theta[:, 0, 1] = -angle.sin(); theta[:, 1, 0] = angle.sin()
    theta[:, :, 2] = (r[:, 1:] * 2 - 1) * (4 / 28)
    grid = F.affine_grid(theta, x.size(), align_corners=False)
    return F.grid_sample(x, grid, mode='bilinear', padding_mode='zeros', align_corners=False)

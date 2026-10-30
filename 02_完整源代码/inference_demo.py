"""从本机训练的最佳权重重新推理，导出效果页及未经处理的原始测试图片。"""
import base64
import csv
import hashlib
import html
import io
import json
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
from pathlib import Path
from layout import BUNDLE, CHECKPOINTS, RESULTS, HISTORIES, PREDICTIONS, FIGURES, VECTOR, DEMO, DATA
import numpy as np
from PIL import Image
import torch
from dataset import load_raw, normalize, CLASSES_CN
from models import ClothingCNN
from train import set_seed

ROOT=Path(__file__).resolve().parent


def main():
    cfg=json.loads((ROOT/'configs/experiment.json').read_text());seed=cfg['primary_seed']
    set_seed(seed)
    out=DEMO;raw=out/'原始输入PNG';raw.mkdir(parents=True,exist_ok=True)
    _,_,images,target,_=load_raw(DATA)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    predictions={};confidence={};hashes={}
    for v,name in [('A','cnn_baseline'),('D','cnn_se_aug')]:
        path=CHECKPOINTS/f'{name}.pth';ckpt=torch.load(path,map_location='cpu',weights_only=True)
        model=ClothingCNN(ckpt['use_se'],cfg['reduction']).to(device);model.load_state_dict(ckpt['model_state']);model.eval()
        x=torch.from_numpy(images).float().unsqueeze(1).to(device)/255
        x=normalize(x,ckpt['normalization']['mean'],ckpt['normalization']['std'])
        with torch.inference_mode():
            prob=torch.cat([model(x[s:s+512]).softmax(1) for s in range(0,len(x),512)]).cpu().numpy()
        predictions[v]=prob.argmax(1);confidence[v]=prob.max(1)
        saved=np.load(PREDICTIONS/f'{name}_seed{seed}.npz')
        assert np.array_equal(saved['prediction'],predictions[v]),'重新推理与保存预测不一致'
        hashes[v]=hashlib.sha256(path.read_bytes()).hexdigest()
    groups={
      'fig13_fixed_predictions':('固定顺序预测效果','官方测试集前 20 张，不按结果挑选。',np.arange(20)),
      'fig14_corrected_predictions':('组合模型纠正的样例','按测试索引选取前 8 个 A 分错、D 分对的样例，仅作定性展示。',np.flatnonzero((predictions['A']!=target)&(predictions['D']==target))[:8]),
      'fig15_failed_predictions':('组合模型仍然分错的样例','按测试索引选取 D 分错的前 8 个样例，展示实际失败情况。',np.flatnonzero(predictions['D']!=target)[:8]),
      'fig16_regressed_predictions':('组合模型退化的样例','按测试索引选取前 8 个 A 分对、D 分错的样例，不隐藏改进的代价。',np.flatnonzero((predictions['A']==target)&(predictions['D']!=target))[:8])
    }
    records=[];pages=[]
    for stem,(title,desc,ids) in groups.items():
        cards=[]
        for ix in ids.tolist():
            image_path=raw/f'test_{ix:05d}.png'
            Image.fromarray(images[ix]).save(image_path)
            encoded=base64.b64encode(image_path.read_bytes()).decode()
            cards.append(f'<article><div class="index">测试样本 #{ix:05d}</div><img src="data:image/png;base64,{encoded}" alt="测试图像 {ix}"><div class="truth">真实类别：{CLASSES_CN[target[ix]]}</div>'+''.join(f'<div class="pred {"ok" if predictions[v][ix]==target[ix] else "bad"}">{v}：{CLASSES_CN[predictions[v][ix]]}　{confidence[v][ix]*100:.1f}%　{"✓" if predictions[v][ix]==target[ix] else "×"}</div>' for v in ['A','D'])+'</article>')
            records.append({'group':stem,'test_index':ix,'true_label':int(target[ix]),'true_class':CLASSES_CN[target[ix]],'A_predicted_label':int(predictions['A'][ix]),'D_predicted_label':int(predictions['D'][ix]),'A_confidence':float(confidence['A'][ix]),'D_confidence':float(confidence['D'][ix]),'original_png':str(image_path.relative_to(BUNDLE))})
        cols=4
        doc='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>'+title+'</title><style>body{margin:0;background:white;color:#152535;font-family:"Microsoft YaHei",sans-serif}main{padding:32px 40px}h1{font-size:36px;margin:0 0 10px}p{font-size:24px;color:#455567;margin:8px 0 22px}.grid{display:grid;grid-template-columns:repeat('+str(cols)+',1fr);gap:18px}article{border:1px solid #ccd5de;padding:14px;text-align:center}.index{font-size:22px;color:#526272;margin-bottom:10px}img{width:112px;height:112px;image-rendering:pixelated;background:black}.truth{font-size:26px;font-weight:600;margin:10px 0}.pred{font-size:26px;padding:5px 0}.ok{color:#254f78}.bad{color:#a24b21}footer{font-size:18px;line-height:1.7;margin-top:22px;color:#586575;word-break:break-all}</style><main><h1>'+title+'</h1><p>'+desc+' A＝基础 CNN；D＝CNN＋SE＋增强；均使用本机真实训练的 seed '+str(seed)+' 最佳验证权重。</p><div class="grid">'+''.join(cards)+'</div><footer>显示百分比为模型 Softmax 最大输出，未经置信度校准。图片是对应测试集的原始 28×28 灰度图，仅在页面按整数倍放大。<br>A 权重 SHA256：'+hashes['A']+'<br>D 权重 SHA256：'+hashes['D']+'</footer></main></html>'
        path=out/f'{stem}.html';path.write_text(doc,encoding='utf-8');pages.append({'stem':stem,'html':str(path),'ids':ids.tolist()})
    (out/'真实预测数据.json').write_text(json.dumps({'seed':seed,'checkpoint_sha256':hashes,'records':records},ensure_ascii=False,indent=2),encoding='utf-8')
    with (out/'真实预测数据.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=records[0].keys());w.writeheader();w.writerows(records)
    (out/'pages.json').write_text(json.dumps(pages,ensure_ascii=False,indent=2),encoding='utf-8')
    print('重新推理与原测试预测一致，已导出真实效果页和原始输入图片。',flush=True)


if __name__=='__main__':main()

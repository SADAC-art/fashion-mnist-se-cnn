"""从真实保存的 JSON/NPZ 生成论文插图，不运行或模拟训练。"""
import json
from pathlib import Path
from layout import BUNDLE, CHECKPOINTS, RESULTS, HISTORIES, PREDICTIONS, FIGURES, VECTOR, DEMO, DATA
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch
from dataset import load_raw, CLASSES_CN
from train import VARIANTS

ROOT = Path(__file__).resolve().parent
FIG = FIGURES
COLORS = {'A':'#23618D','B':'#CD7B22','C':'#8C688B','D':'#687A35'}
NAMES = {'A':'A 基础 CNN','B':'B CNN＋增强','C':'C CNN＋SE','D':'D CNN＋SE＋增强'}


def setup():
    FIG.mkdir(parents=True,exist_ok=True)
    VECTOR.mkdir(parents=True,exist_ok=True)
    for file in [Path('C:/Windows/Fonts/simhei.ttf'),Path('C:/Windows/Fonts/msyh.ttc')]:
        if file.exists():
            font_manager.fontManager.addfont(str(file))
            plt.rcParams['font.family'] = font_manager.FontProperties(fname=str(file)).get_name()
            break
    plt.rcParams.update({'font.size':11,'axes.titlesize':13,'axes.labelsize':11,
                         'axes.unicode_minus':False,'axes.spines.top':False,'axes.spines.right':False,
                         'legend.frameon':False,'savefig.dpi':300,'pdf.fonttype':42})


def save(fig, name):
    fig.savefig(FIG/f'{name}.png', dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(VECTOR/f'{name}.pdf', bbox_inches='tight', facecolor='white')
    plt.close(fig)


def box(ax, x, y, text, width=1.6, height=.75, color='#E9F0F5'):
    ax.add_patch(FancyBboxPatch((x-width/2,y-height/2),width,height,boxstyle='round,pad=0.04,rounding_size=0.06',linewidth=.9,edgecolor='#344450',facecolor=color))
    ax.text(x,y,text,ha='center',va='center',fontsize=10)


def arrow(ax,p1,p2):
    ax.annotate('', xy=p2, xytext=p1, arrowprops={'arrowstyle':'->','lw':1.2,'color':'#344450'})


def network_diagram(use_se, filename):
    fig,ax=plt.subplots(figsize=(10.8,4.2)); ax.set(xlim=(0,11),ylim=(-.1,4.1));ax.axis('off')
    labels = ['输入\n1×28×28','Conv 3×3\n1→32，BN，ReLU',('SE 32→4→32\n通道重标定' if use_se else '恒等映射\n无注意力'),'MaxPool 2×2\n32×14×14',
              'Conv 3×3\n32→64，BN，ReLU',('SE 64→8→64\n通道重标定' if use_se else '恒等映射\n无注意力'),'MaxPool 2×2\n64×7×7','Flatten → FC\n3136→128','ReLU＋Dropout\np＝0.25','FC 128→10\n输出 logits']
    coords=[(1.1+i*2.2,3) for i in range(5)]+[(9.9-i*2.2,1) for i in range(5)]
    for i,(label,(x,y)) in enumerate(zip(labels,coords)):
        box(ax,x,y,label,width=1.85,color='#F2F0E3' if use_se and i in [2,5] else '#E9F0F5')
        if i:
            px,py=coords[i-1]
            if py==y:
                direction=1 if x>px else -1
                arrow(ax,(px+direction*.97,py),(x-direction*.97,y))
            else: arrow(ax,(px,py-.42),(x,y+.42))
    ax.set_title('改进 CNN 结构及特征尺寸' if use_se else '基础 CNN 结构及特征尺寸',pad=8)
    save(fig,filename)


def se_diagram():
    fig,ax=plt.subplots(figsize=(10.8,3.5));ax.axis('off');ax.set(xlim=(0,11),ylim=(0,3.8))
    labels=['输入特征 U\nC×H×W','全局平均池化\nz∈R^C','FC＋ReLU\nC→C/r','FC＋Sigmoid\nC/r→C','逐通道相乘\n输出重标定特征']
    xs=[1.05,3.25,5.45,7.65,9.85]
    for i,(x,label) in enumerate(zip(xs,labels)):
        box(ax,x,1.6,label,width=1.8)
        if i:arrow(ax,(xs[i-1]+.95,1.6),(x-.95,1.6))
    ax.plot([1.05,1.05,9.85],[2.02,2.85,2.85],color='#344450',lw=1)
    arrow(ax,(9.85,2.85),(9.85,2.02));ax.text(5.45,3.0,'保留原始特征，供通道权重重标定',ha='center',fontsize=10)
    ax.set_title('SE 注意力模块流程  本项目 r＝8',pad=8);save(fig,'fig03_se_module')


def main():
    setup();cfg=json.loads((ROOT/'configs/experiment.json').read_text())
    seeds=cfg['seeds'];epochs=np.arange(1,cfg['epochs']+1)
    network_diagram(False,'fig02_cnn_baseline');se_diagram();network_diagram(True,'fig04_cnn_se')
    x,y,_,_,_=load_raw(DATA);idx=np.load(RESULTS / 'split_indices.npz')['train']
    from PIL import Image
    originals=DEMO / '训练样例原图PNG';originals.mkdir(parents=True,exist_ok=True)
    fig,axes=plt.subplots(3,10,figsize=(11.5,4.4))
    for c in range(10):
        chosen=idx[y[idx]==c][:3]
        for row,ix in enumerate(chosen):
            Image.fromarray(x[ix]).save(originals/f'train_{ix:05d}_class{c}.png')
            axes[row,c].imshow(x[ix],cmap='gray',vmin=0,vmax=255);axes[row,c].axis('off')
            if row==0:axes[row,c].set_title(CLASSES_CN[c],fontsize=10)
    fig.suptitle('Fashion-MNIST 训练子集样例  每类展示 3 张',fontsize=13)
    fig.tight_layout();save(fig,'fig01_dataset_samples')
    histories={v: [json.loads((HISTORIES/f'{name}_seed{s}.json').read_text()) for s in seeds] for v,(name,_,_) in VARIANTS.items()}
    fig,ax=plt.subplots(figsize=(8.8,4.6))
    for v in VARIANTS:
        z=np.array([[r['train_loss'] for r in h] for h in histories[v]])
        ax.plot(epochs,z.mean(0),label=NAMES[v],color=COLORS[v]);ax.fill_between(epochs,z.mean(0)-z.std(0,ddof=1),z.mean(0)+z.std(0,ddof=1),alpha=.12,color=COLORS[v])
    ax.set(xlabel='训练轮次 / epoch',ylabel='训练交叉熵损失',title='四组模型训练损失  3 个种子均值与标准差',xticks=[1,5,10,15,20]);ax.grid(axis='y',alpha=.2);ax.legend(ncol=2);fig.tight_layout();save(fig,'fig05_train_loss')
    fig,axes=plt.subplots(2,2,figsize=(9.6,6.4),sharex=True,sharey=True)
    for ax,v in zip(axes.flat,VARIANTS):
        for key,label,style in [('train_accuracy','训练','--'),('val_accuracy','验证','-')]:
            z=np.array([[r[key]*100 for r in h] for h in histories[v]])
            ax.plot(epochs,z.mean(0),style,label=label,color=COLORS[v]);ax.fill_between(epochs,z.mean(0)-z.std(0,ddof=1),z.mean(0)+z.std(0,ddof=1),alpha=.12,color=COLORS[v])
        ax.set(title=NAMES[v],ylim=(70,100),xticks=[1,5,10,15,20]);ax.grid(axis='y',alpha=.2);ax.legend(loc='lower right')
    for ax in axes[-1]:ax.set_xlabel('训练轮次 / epoch')
    for ax in axes[:,0]:ax.set_ylabel('准确率 / %')
    fig.suptitle('训练与验证准确率  3 个种子均值与标准差');fig.tight_layout();save(fig,'fig06_accuracy_curves')
    rows=json.loads((RESULTS / 'all_metrics.json').read_text());summary=json.loads((RESULTS / 'summary.json').read_text())
    fig,ax=plt.subplots(figsize=(8.8,4.6));pos=np.arange(4);means=[r['accuracy_mean']*100 for r in summary];sds=[r['accuracy_std']*100 for r in summary]
    ax.bar(pos,means,yerr=sds,color=[COLORS[v] for v in VARIANTS],capsize=4,width=.56)
    for i,(m,s) in enumerate(zip(means,sds)):ax.text(i,m+s+.7,f'{m:.2f}±{s:.2f}',ha='center',fontsize=10)
    ax.set(xticks=pos,xticklabels=[NAMES[v] for v in VARIANTS],ylim=(0,102),ylabel='测试准确率 / %',title='最终测试准确率  3 个种子均值±样本标准差');ax.grid(axis='y',alpha=.18);fig.tight_layout();save(fig,'fig07_test_accuracy')
    fig,ax=plt.subplots(figsize=(8.8,4.4));values=[r['parameters']/1000 for r in summary];ax.bar(pos,values,color=[COLORS[v] for v in VARIANTS],width=.56)
    for i,m in enumerate(values):ax.text(i,m+5,f'{m:.3f}',ha='center')
    ax.set(xticks=pos,xticklabels=[NAMES[v] for v in VARIANTS],ylim=(0,max(values)*1.16),ylabel='可训练参数量 / 千个',title='模型参数量对比');ax.grid(axis='y',alpha=.18);fig.tight_layout();save(fig,'fig08_parameters')
    matrices={}
    for v,name in [('A','fig09_confusion_A'),('D','fig10_confusion_D')]:
        model_name=VARIANTS[v][0];pred=np.load(PREDICTIONS/f'{model_name}_seed{cfg["primary_seed"]}.npz');cm=pred['confusion'];matrices[v]=cm
        fig,ax=plt.subplots(figsize=(7.7,6.5));im=ax.imshow(cm,cmap='Blues',vmin=0,vmax=1000)
        for i in range(10):
            for j in range(10):ax.text(j,i,str(cm[i,j]),ha='center',va='center',fontsize=8,color='white' if cm[i,j]>550 else '#152C40')
        ax.set(xticks=range(10),yticks=range(10),xticklabels=CLASSES_CN,yticklabels=CLASSES_CN,xlabel='预测类别',ylabel='真实类别',title=f'{NAMES[v]} 混淆矩阵  seed＝{cfg["primary_seed"]}')
        plt.setp(ax.get_xticklabels(),rotation=45,ha='right');fig.colorbar(im,ax=ax,label='样本数 / 张',shrink=.82);fig.tight_layout();save(fig,name)
    fig,ax=plt.subplots(figsize=(10,5.1));pos=np.arange(10)
    for k,(v,cm) in enumerate(matrices.items()):
        values=cm.diagonal()/cm.sum(1)*100
        ax.bar(pos+(k-.5)*.35,values,width=.35,color=COLORS[v],label=NAMES[v])
        for i,m in enumerate(values):ax.text(i+(k-.5)*.35,m+.7,f'{m:.1f}',ha='center',rotation=90,fontsize=8)
    ax.set(xticks=pos,xticklabels=CLASSES_CN,ylim=(0,112),ylabel='各类召回率 / %',title=f'各类别分类准确率对比  seed＝{cfg["primary_seed"]}，每类 1000 张');ax.legend(ncol=2,loc='upper center',bbox_to_anchor=(.5,-.12));fig.tight_layout();save(fig,'fig11_class_recall')
    fig,ax=plt.subplots(figsize=(8.8,4.5));pairs=[('B','A'),('C','A'),('D','B'),('D','C')]
    labels=['增强效应 B-A','SE 效应 C-A','增强下 SE 效应 D-B','SE 下增强效应 D-C']
    deltas=[]
    for left,right in pairs:
        deltas.append([(next(r['accuracy'] for r in rows if r['variant']==left and r['seed']==s)-next(r['accuracy'] for r in rows if r['variant']==right and r['seed']==s))*100 for s in seeds])
    for i,z in enumerate(deltas):
        ax.scatter(np.array([-.1,0,.1])+i,z,color='#23618D',s=35)
        ax.plot([i-.2,i+.2],[np.mean(z)]*2,color='#344450',lw=2)
    ax.axhline(0,color='#344450',ls='--',lw=1);ax.set(xticks=range(4),xticklabels=labels,ylabel='测试准确率差 / 百分点',title='配对种子的模块与增强效应  点为单次，横线为均值');ax.grid(axis='y',alpha=.2);fig.tight_layout();save(fig,'fig12_ablation_deltas')
    (FIG/'figure_manifest.json').write_text(json.dumps([{'file':p.name,'dpi':300} for p in sorted(FIG.glob('*.png'))],ensure_ascii=False,indent=2),encoding='utf-8')
    print('已生成 12 组 PNG/PDF 图像',flush=True)


if __name__=='__main__':main()

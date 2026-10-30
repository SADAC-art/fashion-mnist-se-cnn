"""提交材料统一路径；训练、测试和绘图共用同一份权重与结果。"""
from pathlib import Path
SOURCE = Path(__file__).resolve().parent
BUNDLE = SOURCE.parent
CHECKPOINTS = BUNDLE / '03_模型权重'
RESULTS = BUNDLE / '04_实验结果'
HISTORIES = RESULTS / '训练日志'
PREDICTIONS = RESULTS / '分类结果'
FIGURES = BUNDLE / '05_实验图表' / '原图PNG300DPI'
VECTOR = BUNDLE / '05_实验图表' / '矢量PDF'
DEMO = BUNDLE / '05_实验图表' / '真实预测效果'
DATA = SOURCE / 'data'

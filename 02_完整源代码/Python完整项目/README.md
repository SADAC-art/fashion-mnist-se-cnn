# 运行源码

在本文件夹打开终端，执行：

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe run_all.py
```

程序从同级提交目录的 `03_模型权重`、`04_实验结果` 读取已完成实验。测试单个权重可执行：

```powershell
.\.venv\Scripts\python.exe test.py --checkpoint "../../03_模型权重/cnn_baseline.pth"
```

数据集缺失时自动从官方来源下载。原始训练种子为2026、2027、2028，配置见 `configs/experiment.json`。绘图运行 `visualize.py`；真实效果截图依次运行 `inference_demo.py` 和 `capture_screenshots.py`。重新训练或评价前请复制一份提交材料，保留原始实验记录。

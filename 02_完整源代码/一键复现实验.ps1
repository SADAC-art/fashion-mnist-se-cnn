$ErrorActionPreference='Stop'
Set-Location -LiteralPath $PSScriptRoot
$env:PYTHONIOENCODING='utf-8'
$env:CUBLAS_WORKSPACE_CONFIG=':4096:8'
$projectPython=Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $projectPython)) {
    py -3.13 -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw '创建虚拟环境失败，请安装 Python 3.13。' }
    & $projectPython -m pip install -r requirements.txt
    if ($LASTEXITCODE -ne 0) { throw '安装依赖失败。' }
}
# 若已有完整实验，run_all 会保留20轮训练结果，再从权重执行测试。
& $projectPython -u run_all.py
if ($LASTEXITCODE -ne 0) { throw '实验失败，请查看终端信息。' }
& $projectPython visualize.py
if ($LASTEXITCODE -ne 0) { throw '图表生成失败。' }
& $projectPython inference_demo.py
if ($LASTEXITCODE -ne 0) { throw '真实预测样例生成失败。' }
& $projectPython capture_screenshots.py
if ($LASTEXITCODE -ne 0) { throw '效果页截图失败，请检查本机 Microsoft Edge。' }
Write-Host '实验、图表、原始输入图片和效果截图已生成。' -ForegroundColor Cyan

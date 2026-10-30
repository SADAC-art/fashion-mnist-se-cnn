$ErrorActionPreference='Stop'
Set-Location -LiteralPath $PSScriptRoot
$env:PYTHONIOENCODING='utf-8'
$env:CUBLAS_WORKSPACE_CONFIG=':4096:8'
$projectPython=Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $projectPython)) { throw '先运行一键复现实验脚本以配置环境。' }
& $projectPython -u run_all.py --stage evaluate
if ($LASTEXITCODE -ne 0) { throw '测试失败。' }

# StreamETF 本地启动脚本
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not (Test-Path ".venv")) {
    Write-Host "创建虚拟环境..."
    python -m venv .venv
}

Write-Host "激活虚拟环境并安装依赖..."
& .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt -q

Write-Host ""
Write-Host "启动 StreamETF: http://localhost:8501"
Write-Host "ETFirst CLI 已随 etfirst-0.2.3.tar.gz 安装"
Write-Host "首次使用请运行: .\setup_etfirst.ps1 -ApiKey <你的KEY>"
Write-Host ""

streamlit run app.py --server.port 8501

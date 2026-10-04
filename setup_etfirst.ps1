# ETFirst 登录脚本（需先在首趋E指小程序获取 API Key）
param(
    [Parameter(Mandatory = $true)]
    [string]$ApiKey
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

& .\.venv\Scripts\etfirst.exe auth login --api-key $ApiKey
Write-Host ""
Write-Host "验证配置..."
& .\.venv\Scripts\etfirst.exe config
Write-Host ""
Write-Host "测试查询..."
& .\.venv\Scripts\etfirst.exe --json index-base clas --type 1 | Select-Object -First 5

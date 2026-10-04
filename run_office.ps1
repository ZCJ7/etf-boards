# 公司电脑启动：监听所有网卡，方便同 WiFi 手机访问
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not (Test-Path ".venv")) {
    Write-Host "创建虚拟环境..."
    python -m venv .venv
}

Write-Host "激活虚拟环境并安装依赖..."
& .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt -q

$ips = Get-NetIPAddress -AddressFamily IPv4 -ErrorAction SilentlyContinue |
    Where-Object {
        $_.IPAddress -notlike "127.*" -and
        $_.PrefixOrigin -ne "WellKnown"
    } |
    Select-Object -ExpandProperty IPAddress -Unique

Write-Host ""
Write-Host "StreamETF 已按局域网模式启动（0.0.0.0:8501）"
Write-Host "本机:     http://127.0.0.1:8501"
if ($ips) {
    foreach ($ip in $ips) {
        Write-Host "同 WiFi:  http://${ip}:8501"
    }
} else {
    Write-Host "未读到局域网 IP，可在公司电脑运行: ipconfig"
}
Write-Host ""
Write-Host "手机不在公司 WiFi（4G/家里）时，需要再开穿透（cpolar / Cloudflare Tunnel）。"
Write-Host "首次请确认 Windows 防火墙允许 8501 入站。"
Write-Host ""

streamlit run app.py --server.port 8501 --server.address 0.0.0.0 --server.headless true

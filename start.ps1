# ProRag — start the whole stack (postgres + FastAPI backend + Next.js frontend).
# Usage: powershell -File D:\ragPro\start.ps1
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

Write-Host "1/3 postgres (docker)..."
docker compose up -d postgres | Out-Null

Write-Host "2/3 backend (uvicorn :8000)..."
$up = Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue
if (-not $up) {
  Start-Process -WindowStyle Hidden .\.venv\Scripts\python.exe -ArgumentList "-m","uvicorn","prorag.main:app","--port","8000"
}

Write-Host "3/3 frontend (next dev :3001)..."
$up = Get-NetTCPConnection -LocalPort 3001 -State Listen -ErrorAction SilentlyContinue
if (-not $up) {
  Start-Process -WindowStyle Hidden cmd -ArgumentList "/c","cd /d $PSScriptRoot\web-next && npm run dev"
}

Start-Sleep 6
Write-Host ""
Write-Host "ProRag is up:  http://localhost:3001" -ForegroundColor Green
Write-Host "  backend API: http://127.0.0.1:8000  (docs at /docs)"

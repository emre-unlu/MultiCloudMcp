#Requires -Version 7
$ErrorActionPreference = "Stop"
$PSStyle.OutputRendering = "Ansi"

# Resolve paths from script location
$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$K8sDir = Join-Path $RepoRoot "mcp\kubernetes"
$SupDir = Join-Path $RepoRoot "supervisor"
$AppDir = Join-Path $RepoRoot "app"

# Start K8s MCP Server
Write-Host "Starting Kubernetes MCP Server..." -ForegroundColor Yellow
$k8sCmd = @"
Set-Location '$K8sDir'
& '${K8sDir}\.venv\Scripts\python.exe' server.py
Read-Host "Press Enter to exit"
"@
Start-Process -WindowStyle Normal -FilePath "powershell" -ArgumentList "-ExecutionPolicy Bypass -Command", $k8sCmd

# Start Supervisor (OpenAI API + chatgpt mini example)
Write-Host "Starting Supervisor..." -ForegroundColor Yellow
$supCmd = @"
Set-Location '$RepoRoot'
`$env:MODEL = 'gpt-5-mini-2025-08-07'
`$env:DIAGNOSTICS_MODEL = 'gpt-5-mini-2025-08-07'
if (-not `$env:OPENAI_API_KEY) {
  Write-Host 'WARNING: OPENAI_API_KEY is not set in this terminal/session.' -ForegroundColor Yellow
}
& '${SupDir}\.venv\Scripts\uvicorn.exe' supervisor.app:app --reload --port 9000
Read-Host "Press Enter to exit"
"@
Start-Process -WindowStyle Normal -FilePath "powershell" -ArgumentList "-ExecutionPolicy Bypass -Command", $supCmd

# Start Go UI Server
Write-Host "Starting UI Server..." -ForegroundColor Yellow
$uiCmd = @"
Set-Location '$AppDir'
go run .
Read-Host "Press Enter to exit"
"@
Start-Process -WindowStyle Normal -FilePath "powershell" -ArgumentList "-Command", $uiCmd

Write-Host "`n✅ Services started:" -ForegroundColor Green
Write-Host "- Kubernetes MCP Server running in new window"
Write-Host "- Supervisor running in new window on http://127.0.0.1:9000 (MODEL=gpt-5-mini-2025-08-07)"
Write-Host "- UI Server running in new window on http://127.0.0.1:8088"

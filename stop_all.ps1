# Stop common dev processes we started
$names = @("uvicorn", "python", "ollama")
foreach ($n in $names) {
  $ps = Get-Process -ErrorAction SilentlyContinue | Where-Object { $_.ProcessName -like "*$n*" }
  foreach ($p in $ps) {
    try {
      Write-Host ("Stopping {0} (PID {1})..." -f $p.ProcessName, $p.Id) -ForegroundColor Yellow
      $p.Kill()
    } catch { }
  }
}
Write-Host "Done."

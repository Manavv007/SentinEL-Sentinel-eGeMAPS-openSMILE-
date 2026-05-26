# Stop anything on port 8765 and start SentinEL web UI fresh.
$ErrorActionPreference = "SilentlyContinue"
Set-Location $PSScriptRoot

$conns = Get-NetTCPConnection -LocalPort 8765 -State Listen -ErrorAction SilentlyContinue
foreach ($c in $conns) {
    $procId = $c.OwningProcess
    Write-Host "Stopping process on port 8765 (PID $procId)..."
    Stop-Process -Id $procId -Force -ErrorAction SilentlyContinue
}
Start-Sleep -Seconds 1

& "$PSScriptRoot\run_web.ps1"

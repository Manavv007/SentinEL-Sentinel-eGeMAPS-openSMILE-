# Start SentinEL web UI (http://127.0.0.1:8765)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

. "$PSScriptRoot\scripts\resolve_python.ps1"

$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONPATH = $PSScriptRoot

Write-Host ""
Write-Host "SentinEL Web UI -> http://127.0.0.1:8765"
Write-Host ""

if ($SentinelPython -match "^py ") {
    Invoke-Expression "$SentinelPython -m uvicorn web.app:app --host 127.0.0.1 --port 8765 --app-dir `"$PSScriptRoot`""
} else {
    & $SentinelPython -m uvicorn web.app:app --host 127.0.0.1 --port 8765 --app-dir $PSScriptRoot
}

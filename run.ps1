# SentinEL CLI — uses Python with whisperx installed.
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

. "$PSScriptRoot\scripts\resolve_python.ps1"

$env:PYTHONIOENCODING = "utf-8"

if ($SentinelPython -match "^py ") {
    Invoke-Expression "$SentinelPython main.py @args"
} else {
    & $SentinelPython main.py @args
}

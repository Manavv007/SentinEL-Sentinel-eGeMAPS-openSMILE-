# Resolve Python executable that has SentinEL dependencies (whisperx).
# Sets $SentinelPython to the chosen path.

$SentinelPython = $env:SENTINEL_PYTHON
$envFile = Join-Path (Split-Path $PSScriptRoot -Parent) ".env"
if (-not $SentinelPython -and (Test-Path $envFile)) {
    Get-Content $envFile | ForEach-Object {
        if ($_ -match '^\s*SENTINEL_PYTHON\s*=\s*(.+)\s*$') {
            $SentinelPython = $Matches[1].Trim().Trim('"').Trim("'")
        }
    }
}

$candidates = @(
    $SentinelPython
    "$env:LOCALAPPDATA\Microsoft\WindowsApps\PythonSoftwareFoundation.Python.3.13_qbz5n2kfra8p0\python.exe"
    "$env:LOCALAPPDATA\Programs\Python\Python313\python.exe"
    "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe"
)

try {
    $pyLauncher = (Get-Command py -ErrorAction SilentlyContinue)
    if ($pyLauncher) {
        $candidates += @("py -3.13", "py -3.12")
    }
} catch {}

$candidates += @("python3", "python")

function Test-SentinelPython([string]$exe) {
    if (-not $exe) { return $false }
    if ($exe -match "^py ") {
        $out = Invoke-Expression "$exe -c `"import whisperx`"" 2>&1
    } elseif (Test-Path $exe) {
        $out = & $exe -c "import whisperx" 2>&1
    } else {
        return $false
    }
    return $LASTEXITCODE -eq 0
}

$chosen = $null
foreach ($c in $candidates) {
    if (-not $c) { continue }
    if (Test-SentinelPython $c) {
        $chosen = $c
        break
    }
}

if (-not $chosen) {
    Write-Host ""
    Write-Host "ERROR: No Python with 'whisperx' found." -ForegroundColor Red
    Write-Host ""
    Write-Host "Install dependencies (from project README), then retry:"
    Write-Host "  pip install torch==2.3.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cpu"
    Write-Host "  pip install whisperx==3.1.5"
    Write-Host "  pip install -r requirements.txt"
    Write-Host ""
    Write-Host "Or set SENTINEL_PYTHON to your python.exe path in .env"
    Write-Host ""
    exit 1
}

$SentinelPython = $chosen
Write-Host "Using Python: $SentinelPython"

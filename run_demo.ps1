param([int]$Port = 8501)
$ErrorActionPreference = 'Stop'
if ($Port -lt 1024 -or $Port -gt 65535) { throw 'Port must be between 1024 and 65535.' }
Set-Location -LiteralPath $PSScriptRoot
$demoPythonPath = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $demoPythonPath)) {
    $created = $false
    try {
        & py -3.11 -m venv (Join-Path $PSScriptRoot '.venv')
        if ($LASTEXITCODE -eq 0) { $created = $true }
    } catch { }
    if (-not $created) {
        & python -m venv (Join-Path $PSScriptRoot '.venv')
        if ($LASTEXITCODE -ne 0) { throw 'Cannot create Python environment. Install Python 3.11 64-bit.' }
    }
}
& $demoPythonPath -c "import json; from importlib.metadata import version; from pathlib import Path; expected=json.loads(Path('demo/artifacts/model_registry.json').read_text(encoding='utf-8'))['runtime_versions']; expected['streamlit']='1.65.0'; assert all(version(k)==v for k,v in expected.items())" 2>$null
if ($LASTEXITCODE -ne 0) {
    & $demoPythonPath -m pip install -r (Join-Path $PSScriptRoot 'demo\requirements.txt')
    if ($LASTEXITCODE -ne 0) { throw 'Cannot install demo dependencies.' }
}
if (Test-Path -LiteralPath (Join-Path $PSScriptRoot 'verify_bundle.py')) {
    & $demoPythonPath (Join-Path $PSScriptRoot 'verify_bundle.py')
    if ($LASTEXITCODE -ne 0) { throw 'Bundle checksum verification failed.' }
}
& $demoPythonPath -m streamlit run (Join-Path $PSScriptRoot 'demo\app.py') --server.address 127.0.0.1 --server.port $Port --server.headless true --browser.gatherUsageStats false
if ($LASTEXITCODE -ne 0) { throw 'Demo exited with an error.' }

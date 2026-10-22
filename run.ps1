param([Parameter(ValueFromRemainingArguments = $true)] $Args)

$python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    $python = "python"
}

Push-Location $PSScriptRoot
try {
    & $python detect.py @Args
} finally {
    Pop-Location
}

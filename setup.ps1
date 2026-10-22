$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

Write-Host "Создаю виртуальное окружение .venv ..."
py -3 -m venv .venv
$python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"

Write-Host "Обновляю pip ..."
& $python -m pip install --upgrade pip

Write-Host "Ставлю зависимости (torch/ultralytics/CLIP качаются один раз, ~1 ГБ) ..."
& $python -m pip install -r requirements.txt

Write-Host ""
Write-Host "Готово. Запуск камеры:"
Write-Host "  .\.venv\Scripts\python.exe detect.py --source 0"
Write-Host "Проверка окружения:"
Write-Host "  .\.venv\Scripts\python.exe detect.py --doctor"

# scripts/cleanup_pycache.ps1
# Removes all __pycache__ directories and .pyc files from the project
# Excludes .venv directory

$ErrorActionPreference = "Continue"

$scriptDir = $PSScriptRoot
if ([string]::IsNullOrEmpty($scriptDir)) {
    $scriptDir = "."
}
# Point to repository root (one level up from scripts/)
$root = Split-Path -Parent (Resolve-Path $scriptDir)

$pycacheCount = 0
$pycCount = 0

Get-ChildItem -Path $root -Recurse -Directory -Filter "__pycache__" -Force | ForEach-Object {
    $pycachePath = $_.FullName
    if (-not ($pycachePath -match "\\\.venv\\")) {
        $count = (Get-ChildItem -Path $pycachePath -Recurse -File | Measure-Object).Count
        Remove-Item -Path $pycachePath -Recurse -Force -ErrorAction SilentlyContinue
        $script:pycacheCount++
        Write-Host "Removed __pycache__ ($count files): $pycachePath" -ForegroundColor Cyan
    }
}

Get-ChildItem -Path $root -Recurse -Filter "*.pyc" -Force | ForEach-Object {
    $pycPath = $_.FullName
    if (-not ($pycPath -match "\\\.venv\\")) {
        Remove-Item -Path $pycPath -Force -ErrorAction SilentlyContinue
        $script:pycCount++
        Write-Host "Removed .pyc: $pycPath" -ForegroundColor DarkGray
    }
}

Write-Host ""
Write-Host "=== Cleanup Complete ===" -ForegroundColor Green
Write-Host "__pycache__ directories removed: $pycacheCount"
Write-Host ".pyc files removed: $pycCount"

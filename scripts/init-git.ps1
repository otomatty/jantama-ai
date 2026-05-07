# Windows PowerShell 用 git 初期化スクリプト
# 使い方:
#   cd C:\Users\saedg\apps\jantama-ai
#   powershell -ExecutionPolicy Bypass -File scripts\init-git.ps1

$ErrorActionPreference = "Stop"

# プロジェクトルートに移動
Set-Location (Split-Path -Parent $PSScriptRoot)

# 既に git 管理下なら何もしない
if (Test-Path .git) {
    Write-Host "既に git リポジトリです。何もしません。" -ForegroundColor Yellow
    exit 0
}

git init -b main
git add .
git commit -m "chore: initial project scaffold (Phase A)"

Write-Host ""
Write-Host "✅ git リポジトリを初期化しました。" -ForegroundColor Green
Write-Host "   GitHub に push する場合は次のコマンドを実行してください:" -ForegroundColor Cyan
Write-Host ""
Write-Host "   git remote add origin git@github.com:<username>/jantama-ai.git" -ForegroundColor Cyan
Write-Host "   git push -u origin main" -ForegroundColor Cyan

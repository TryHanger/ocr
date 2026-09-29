# Document AI Control Center - Safe Self-Update Script (PowerShell)
$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$AppDir = Resolve-Path "$ScriptDir\.."
$RepoDir = Resolve-Path "$AppDir\.."

Write-Host "=== Document AI Control Center Update Starting ===" -ForegroundColor Cyan
Write-Host "App Directory: $AppDir"

# 1. Database backup
Write-Host "--> Creating pre-update database backup..." -ForegroundColor Yellow
$BackupDir = Join-Path $AppDir "backups"
if (-not (Test-Path $BackupDir)) {
    New-Item -ItemType Directory -Path $BackupDir | Out-Null
}
$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$BackupFile = Join-Path $BackupDir "db_backup_$Timestamp.sql"

$RunningContainers = docker ps --format '{{.Names}}'
if ($RunningContainers -match "document_ai_postgres") {
    docker exec document_ai_postgres pg_dump -U postgres document_ai > $BackupFile
    Write-Host "Backup saved to: $BackupFile" -ForegroundColor Green
} else {
    Write-Host "PostgreSQL container is not running; skipping backup." -ForegroundColor Gray
}

# 2. Pull updates
Write-Host "--> Pulling latest code changes..." -ForegroundColor Yellow
Set-Location $RepoDir
git pull

# 3. Build containers
Write-Host "--> Building updated containers..." -ForegroundColor Yellow
Set-Location $AppDir
docker compose build

# 4. Run migrations
Write-Host "--> Running database migrations..." -ForegroundColor Yellow
docker compose run --rm backend alembic upgrade head

# 5. Restart services
Write-Host "--> Restarting services..." -ForegroundColor Yellow
docker compose up -d

# 6. Health check
Write-Host "--> Verifying service health..." -ForegroundColor Yellow
$Attempts = 0
$MaxAttempts = 12
$Healthy = $false

while ($Attempts -lt $MaxAttempts) {
    try {
        $HealthRes = Invoke-RestMethod -Uri "http://localhost:8000/health" -TimeoutSec 3 -ErrorAction SilentlyContinue
        $ReadyRes = Invoke-RestMethod -Uri "http://localhost:8000/ready" -TimeoutSec 3 -ErrorAction SilentlyContinue
        if ($HealthRes.status -eq "healthy" -and $ReadyRes.status -eq "ready") {
            $Healthy = $true
            break
        }
    } catch {
        # Waiting for startup
    }
    $Attempts++
    Write-Host "Waiting for services... ($Attempts/$MaxAttempts)"
    Start-Sleep -Seconds 5
}

if ($Healthy) {
    Write-Host "=== Update successfully completed! System is healthy. ===" -ForegroundColor Green
} else {
    Write-Host "✖ Health check failed after update!" -ForegroundColor Red
    exit 1
}

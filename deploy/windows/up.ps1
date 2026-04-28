# IsiDetector — daily start (Windows, CPU-only).
#
# Mirrors deploy/_impl/up.sh: bring up the Docker stack, wait for the
# Flask web container to become reachable, open the default browser at
# the dashboard URL.
#
# Usage:
#   deploy\windows\up.ps1                 # daily run
#   deploy\windows\up.ps1 -NoBrowser      # start the stack but don't open the browser
#   deploy\windows\up.ps1 -TimeoutSec 600 # extend the readiness probe deadline
#   deploy\windows\up.ps1 -Url http://localhost:9501
#
# Operators normally double-click Start.bat at the repo root, which
# calls this script with -ExecutionPolicy Bypass.

[CmdletBinding()]
param(
    [string]$Url = "http://localhost:9501",
    [int]$TimeoutSec = 300,
    [switch]$NoBrowser
)

$ErrorActionPreference = "Stop"

# Locate the repo and deploy/ from this script's path.
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$RepoRoot  = (Resolve-Path (Join-Path $ScriptDir "..\..")).Path
$DeployDir = (Resolve-Path (Join-Path $ScriptDir "..")).Path
Set-Location $DeployDir

# ── Pre-flight: Docker engine reachable ─────────────────────────────────────
docker info > $null 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "[INFO] Docker engine not reachable — starting Docker Desktop..." -ForegroundColor Cyan
    $dockerExe = "$env:ProgramFiles\Docker\Docker\Docker Desktop.exe"
    if (-not (Test-Path $dockerExe)) {
        Write-Host "[FAIL] Docker Desktop not found at the default install path." -ForegroundColor Red
        Write-Host "       Run Install.bat first (or start Docker Desktop manually)." -ForegroundColor Red
        Set-Location $RepoRoot
        exit 1
    }
    Start-Process $dockerExe
    $deadline = (Get-Date).AddSeconds(120)
    do {
        Start-Sleep -Seconds 3
        docker info > $null 2>&1
    } while ($LASTEXITCODE -ne 0 -and (Get-Date) -lt $deadline)
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[FAIL] Docker Desktop did not become ready within 2 minutes." -ForegroundColor Red
        Set-Location $RepoRoot
        exit 1
    }
}
Write-Host "[ OK ] Docker engine reachable" -ForegroundColor Green

# ── Compose profile ─────────────────────────────────────────────────────────
# CPU-only on Windows site PCs by spec. The compose stack pairs the
# bucket-aware compose.yaml with the CPU overlay; same files used on
# Linux site PCs run via deploy/_impl/up.sh.
$ComposeArgs = @("compose", "-f", "docker-compose.yml", "-f", "docker-compose.cpu.yml")
Write-Host "▶ Using CPU compose profile" -ForegroundColor Cyan

# ── Bring up the stack ──────────────────────────────────────────────────────
Write-Host "▶ Starting IsiDetector stack (docker compose up -d --build)..." -ForegroundColor Cyan
& docker @ComposeArgs up -d --build
if ($LASTEXITCODE -ne 0) {
    Write-Host "[FAIL] docker compose up failed." -ForegroundColor Red
    Set-Location $RepoRoot
    exit 1
}

# ── Readiness probe ─────────────────────────────────────────────────────────
# bash up.sh greps for the Flask startup banner in the container logs.
# On Windows we use an HTTP probe instead — the Flask root returns 200
# the moment the app is ready, no log parsing involved.
Write-Host "▶ Waiting for web container (timeout ${TimeoutSec}s)..." -ForegroundColor Cyan
$deadline = (Get-Date).AddSeconds($TimeoutSec)
$ready = $false
while ((Get-Date) -lt $deadline) {
    try {
        $resp = Invoke-WebRequest -Uri $Url -UseBasicParsing -TimeoutSec 2 -ErrorAction Stop
        if ($resp.StatusCode -eq 200) { $ready = $true; break }
    } catch {
        # not ready yet — keep polling
    }
    Start-Sleep -Seconds 2
}
if ($ready) {
    Write-Host "[ OK ] Web container ready" -ForegroundColor Green
} else {
    Write-Host "[WARN] Readiness probe timed out — opening browser anyway" -ForegroundColor Yellow
}

# ── Open the UI ─────────────────────────────────────────────────────────────
if ($NoBrowser) {
    Write-Host "[INFO] -NoBrowser specified — skipping browser launch" -ForegroundColor Cyan
} else {
    Start-Process $Url
    Write-Host "[ OK ] Opened $Url" -ForegroundColor Green
}

Set-Location $RepoRoot

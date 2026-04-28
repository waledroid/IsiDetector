# IsiDetector — one-shot Windows installer (CPU-only site PCs).
#
# What it does:
#   1. Verifies Windows 10 build 19041+ or Windows 11.
#   2. Installs Docker Desktop silently if missing (WSL2 backend).
#   3. Writes %USERPROFILE%\.wslconfig with site-PC defaults
#      (4 GB RAM / 2 vCPU / 2 GB swap) so the hidden WSL VM doesn't
#      starve the host.
#   4. Starts Docker Desktop and waits for the engine.
#   5. Calls deploy\windows\run_start.ps1 to build the CPU image.
#   6. Drops a desktop shortcut to Start.bat.
#
# Operators normally double-click Install.bat at the repo root, which
# elevates and runs this script. Running it from a non-elevated shell
# fails by design — Docker Desktop install needs admin.

#Requires -RunAsAdministrator

[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$RepoRoot  = (Resolve-Path (Join-Path $ScriptDir "..\..")).Path
Set-Location $RepoRoot

Write-Host ""
Write-Host "═══════════════════════════════════════════════════" -ForegroundColor White
Write-Host "  IsiDetector — Windows Installer (CPU-only)" -ForegroundColor White
Write-Host "═══════════════════════════════════════════════════" -ForegroundColor White
Write-Host ""

# ── Stage 1/5: Windows version ──────────────────────────────────────────────
$os = Get-CimInstance Win32_OperatingSystem
$build = [int]$os.BuildNumber
Write-Host "[INFO] Windows: $($os.Caption) (build $build)" -ForegroundColor Cyan
if ($build -lt 19041) {
    Write-Host "[FAIL] Need Windows 10 build 19041+ or Windows 11 for WSL2 + Docker Desktop." -ForegroundColor Red
    Write-Host "       Update Windows, then re-run this installer." -ForegroundColor Red
    exit 1
}
Write-Host "[ OK ] Windows version supported" -ForegroundColor Green

# ── Stage 2/5: Docker Desktop ───────────────────────────────────────────────
$dockerExe = "$env:ProgramFiles\Docker\Docker\Docker Desktop.exe"
if (Test-Path $dockerExe) {
    Write-Host "[ OK ] Docker Desktop already installed" -ForegroundColor Green
} else {
    Write-Host ""
    Write-Host "[INFO] Downloading Docker Desktop installer..." -ForegroundColor Cyan
    $installerUrl  = "https://desktop.docker.com/win/main/amd64/Docker%20Desktop%20Installer.exe"
    $installerPath = Join-Path $env:TEMP "DockerDesktopInstaller.exe"
    try {
        Invoke-WebRequest -Uri $installerUrl -OutFile $installerPath -UseBasicParsing
    } catch {
        Write-Host "[FAIL] Could not download Docker Desktop installer:" -ForegroundColor Red
        Write-Host "       $($_.Exception.Message)" -ForegroundColor Red
        Write-Host "       Check the site PC's internet connection and retry." -ForegroundColor Red
        exit 1
    }
    Write-Host "[ OK ] Installer downloaded ($([math]::Round((Get-Item $installerPath).Length / 1MB)) MB)" -ForegroundColor Green

    Write-Host "[INFO] Running silent install (5-10 minutes; do not interrupt)..." -ForegroundColor Cyan
    $proc = Start-Process -FilePath $installerPath `
        -ArgumentList "install","--quiet","--accept-license","--backend=wsl-2" `
        -Wait -PassThru
    if ($proc.ExitCode -ne 0) {
        Write-Host "[FAIL] Docker Desktop install failed (exit code $($proc.ExitCode))." -ForegroundColor Red
        Write-Host "       The installer log is in %LOCALAPPDATA%\Docker\install-log.txt" -ForegroundColor Red
        exit 1
    }
    Write-Host "[ OK ] Docker Desktop installed" -ForegroundColor Green
    Write-Host ""
    Write-Host "[NOTE] If this is your first Docker install on this account you may need to" -ForegroundColor Yellow
    Write-Host "       log out and back in (or reboot) before Docker Desktop can run. The" -ForegroundColor Yellow
    Write-Host "       installer adds your account to docker-users; group membership only" -ForegroundColor Yellow
    Write-Host "       refreshes at next sign-in." -ForegroundColor Yellow
}

# ── Stage 3/5: .wslconfig ───────────────────────────────────────────────────
$wslConfigPath     = Join-Path $env:USERPROFILE ".wslconfig"
$wslConfigTemplate = Join-Path $ScriptDir "wslconfig.template"
Write-Host ""
if (Test-Path $wslConfigPath) {
    Write-Host "[SKIP] $wslConfigPath already exists — leaving operator's tuning intact" -ForegroundColor Yellow
} else {
    if (-not (Test-Path $wslConfigTemplate)) {
        Write-Host "[FAIL] Missing template: $wslConfigTemplate" -ForegroundColor Red
        Write-Host "       Repo may be incomplete — re-clone the deploy branch." -ForegroundColor Red
        exit 1
    }
    Copy-Item $wslConfigTemplate $wslConfigPath
    Write-Host "[ OK ] Wrote $wslConfigPath (4 GB RAM / 2 vCPU / 2 GB swap)" -ForegroundColor Green
    # If a WSL VM is already up (e.g. from a prior Docker Desktop run),
    # bounce it so the new limits apply on next start.
    & wsl --shutdown 2>$null
}

# ── Stage 4/5: Build the image ──────────────────────────────────────────────
Write-Host ""
Write-Host "[INFO] Starting Docker Desktop (allow up to 3 minutes for the engine)..." -ForegroundColor Cyan
Start-Process $dockerExe
$deadline = (Get-Date).AddSeconds(180)
do {
    Start-Sleep -Seconds 5
    docker info > $null 2>&1
} while ($LASTEXITCODE -ne 0 -and (Get-Date) -lt $deadline)

if ($LASTEXITCODE -ne 0) {
    Write-Host "[FAIL] Docker engine did not become reachable within 3 minutes." -ForegroundColor Red
    Write-Host "       Open Docker Desktop manually, wait for the whale icon to stop animating," -ForegroundColor Red
    Write-Host "       then run: deploy\windows\run_start.ps1" -ForegroundColor Red
    exit 1
}
Write-Host "[ OK ] Docker engine reachable" -ForegroundColor Green

# Run the bootstrap (build + marker + dirs). It exits non-zero on failure.
& powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $ScriptDir "run_start.ps1")
if ($LASTEXITCODE -ne 0) {
    Write-Host "[FAIL] run_start.ps1 reported a build failure — see above." -ForegroundColor Red
    exit 1
}

# ── Stage 5/5: Desktop shortcut ─────────────────────────────────────────────
$startBat = Join-Path $RepoRoot "Start.bat"
if (-not (Test-Path $startBat)) {
    Write-Host "[WARN] Start.bat not found at repo root — skipping desktop shortcut." -ForegroundColor Yellow
} else {
    $desktop      = [Environment]::GetFolderPath("CommonDesktopDirectory")
    if (-not $desktop) { $desktop = [Environment]::GetFolderPath("Desktop") }
    $shortcutPath = Join-Path $desktop "IsiDetector.lnk"
    $wshell       = New-Object -ComObject WScript.Shell
    $shortcut     = $wshell.CreateShortcut($shortcutPath)
    $shortcut.TargetPath       = $startBat
    $shortcut.WorkingDirectory = $RepoRoot
    $shortcut.IconLocation     = "$dockerExe,0"
    $shortcut.Description      = "Start the IsiDetector inference stack"
    $shortcut.Save()
    Write-Host "[ OK ] Desktop shortcut: $shortcutPath" -ForegroundColor Green
}

# ── Done ────────────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "═══════════════════════════════════════════════════" -ForegroundColor White
Write-Host "  Install complete" -ForegroundColor White
Write-Host "═══════════════════════════════════════════════════" -ForegroundColor White
Write-Host ""
Write-Host "  Daily start:    double-click 'IsiDetector' on the desktop"
Write-Host "                  (or run deploy\windows\up.ps1)"
Write-Host ""
Write-Host "  Web UI:         http://localhost:9501"
Write-Host "  Drop weights:   $RepoRoot\isidet\models\"
Write-Host ""
Write-Host "  WSL VM caps:    4 GB RAM / 2 vCPU (edit %USERPROFILE%\.wslconfig to tune)"
Write-Host ""

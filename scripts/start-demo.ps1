# Brings the app up for a local demo: Docker Desktop, the webcam MJPEG
# bridge (so the container can see your laptop's webcam), and the stack.
# Run scripts/stop-demo.ps1 when you're done.

$ProjectDir = Split-Path $PSScriptRoot -Parent
Set-Location $ProjectDir

# ── Docker Desktop ───────────────────────────────────────────────────
$dockerRunning = $false
try { docker info 2>$null | Out-Null; $dockerRunning = $true } catch {}

if (-not $dockerRunning) {
    Write-Host "Starting Docker Desktop..."
    Start-Process "C:\Program Files\Docker\Docker\Docker Desktop.exe"
    Write-Host "Waiting for Docker engine (this takes ~30s on first launch)..."
    Start-Sleep 10
    do {
        Start-Sleep 5
        try { docker info 2>$null | Out-Null; $dockerRunning = $true } catch {}
        if (-not $dockerRunning) { Write-Host "Still waiting..." }
    } while (-not $dockerRunning)
    Write-Host "Docker is ready."
}

# ── Webcam bridge (skip if you're using DroidCam/RTSP instead) ──────
$bridgeUp = $false
try {
    $r = Invoke-WebRequest -Uri http://localhost:8765/health -UseBasicParsing -TimeoutSec 1 -ErrorAction Stop
    $bridgeUp = $true
} catch {}

if (-not $bridgeUp) {
    Write-Host "Starting webcam bridge on :8765..."
    Start-Process python -ArgumentList "$ProjectDir\scripts\webcam_server.py" -WindowStyle Hidden
    Start-Sleep 2
}

# ── App stack ─────────────────────────────────────────────────────────
docker compose up -d

Write-Host ""
Write-Host "Ready: http://localhost"
Write-Host "Stop everything with: scripts\stop-demo.ps1"

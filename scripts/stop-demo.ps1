# Tears down everything start-demo.ps1 brought up.

$ProjectDir = Split-Path $PSScriptRoot -Parent
Set-Location $ProjectDir

docker compose down

# Kill the webcam bridge by the port it listens on
$pid = (netstat -ano | Select-String ":8765 .*LISTENING" | ForEach-Object {
    ($_ -split '\s+')[-1]
} | Select-Object -First 1)

if ($pid) {
    Write-Host "Stopping webcam bridge (PID $pid)..."
    taskkill /F /PID $pid 2>$null
}

Write-Host "Stopped."

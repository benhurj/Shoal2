# start_cluster.ps1
# Launches K Ollama worker instances + L compiler instances
# Usage:
#   .\start_cluster.ps1 start    # Launch all instances
#   .\start_cluster.ps1 stop     # Kill all instances
#   .\start_cluster.ps1 status   # Check health of all instances

param(
    [Parameter(Position=0)]
    [ValidateSet("start", "stop", "status")]
    [string]$Action = "start"
)

# ── Configuration ──
$WorkerModel   = "gemma3:4b"
$CompilerModel = "gemma3:12b"
$WorkerPorts   = @(11434, 11435, 11436, 11437, 11438)  # K=5
$CompilerPorts = @(11440)                                # L=1
$AllPorts      = $WorkerPorts + $CompilerPorts
$OllamaPath    = "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe"

function Start-OllamaInstance {
    param([int]$Port, [string]$Model)

    Write-Host "[*] Starting Ollama on port $Port for model '$Model'..." -ForegroundColor Cyan

    # Each instance needs its own OLLAMA_HOST
    $env:OLLAMA_HOST = "127.0.0.1:$Port"

    # Start ollama serve as a background process
    $job = Start-Process -FilePath $OllamaPath `
        -ArgumentList "serve" `
        -PassThru `
        -WindowStyle Hidden

    Write-Host "    PID: $($job.Id)" -ForegroundColor DarkGray

    # Wait for it to become healthy
    $maxWait = 30
    $waited = 0
    while ($waited -lt $maxWait) {
        Start-Sleep -Seconds 1
        $waited++
        try {
            $response = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/api/tags" -Method Get -ErrorAction Stop
            Write-Host "    Port $Port is healthy!" -ForegroundColor Green
            break
        } catch {
            if ($waited % 5 -eq 0) {
                Write-Host "    Waiting for port $Port... ($waited s)" -ForegroundColor Yellow
            }
        }
    }

    if ($waited -ge $maxWait) {
        Write-Host "    WARNING: Port $Port did not respond within ${maxWait}s" -ForegroundColor Red
        return
    }

    # Pull the model if not already available
    Write-Host "    Pulling model '$Model'..." -ForegroundColor Cyan
    $env:OLLAMA_HOST = "127.0.0.1:$Port"
    & $OllamaPath pull $Model 2>$null
    Write-Host "    Model '$Model' ready on port $Port" -ForegroundColor Green
}

function Stop-AllInstances {
    Write-Host "[*] Stopping all Ollama instances..." -ForegroundColor Yellow
    foreach ($port in $AllPorts) {
        $connections = netstat -ano | Select-String ":$port\s.*LISTENING"
        foreach ($conn in $connections) {
            $procId = ($conn -split '\s+')[-1]
            if ($procId -and $procId -ne "0") {
                try {
                    Stop-Process -Id $procId -Force -ErrorAction Stop
                    Write-Host "    Killed PID $procId (port $port)" -ForegroundColor Green
                } catch {
                    Write-Host "    Could not kill PID $procId (port $port)" -ForegroundColor Red
                }
            }
        }
    }
    Write-Host "[*] All instances stopped." -ForegroundColor Green
}

function Get-ClusterStatus {
    Write-Host "`n=== Cluster Status ===" -ForegroundColor Cyan
    foreach ($port in $WorkerPorts) {
        try {
            $response = Invoke-RestMethod -Uri "http://127.0.0.1:$port/api/tags" -Method Get -ErrorAction Stop -TimeoutSec 2
            $models = ($response.models | ForEach-Object { $_.name }) -join ", "
            Write-Host "  Worker  :$port  UP  [$models]" -ForegroundColor Green
        } catch {
            Write-Host "  Worker  :$port  DOWN" -ForegroundColor Red
        }
    }
    foreach ($port in $CompilerPorts) {
        try {
            $response = Invoke-RestMethod -Uri "http://127.0.0.1:$port/api/tags" -Method Get -ErrorAction Stop -TimeoutSec 2
            $models = ($response.models | ForEach-Object { $_.name }) -join ", "
            Write-Host "  Compiler:$port  UP  [$models]" -ForegroundColor Green
        } catch {
            Write-Host "  Compiler:$port  DOWN" -ForegroundColor Red
        }
    }
    Write-Host ""
}

# ── Main ──
switch ($Action) {
    "start" {
        Write-Host "`n=====================================" -ForegroundColor Cyan
        Write-Host "  Shoal2 LLM Cluster Launcher" -ForegroundColor Cyan
        Write-Host "  Workers:  $($WorkerPorts.Count) x $WorkerModel" -ForegroundColor Cyan
        Write-Host "  Compiler: $($CompilerPorts.Count) x $CompilerModel" -ForegroundColor Cyan
        Write-Host "=====================================`n" -ForegroundColor Cyan

        foreach ($port in $WorkerPorts) {
            Start-OllamaInstance -Port $port -Model $WorkerModel
        }
        foreach ($port in $CompilerPorts) {
            Start-OllamaInstance -Port $port -Model $CompilerModel
        }

        Write-Host "`n[*] Cluster is ready!" -ForegroundColor Green
        Get-ClusterStatus
    }
    "stop" {
        Stop-AllInstances
    }
    "status" {
        Get-ClusterStatus
    }
}

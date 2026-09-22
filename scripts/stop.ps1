# Stops this project's server and tunnel. Nothing else.
#
# Ownership is decided in _common.ps1 by port and command line, so a python.exe
# or cloudflared.exe belonging to another project is never touched -- including
# a cloudflared serving a different local port.
#
# The server's real process is a CHILD of what Start-Process returns, so this
# stops the whole owned tree and then waits for the port to actually clear.
# Returning before the port is free is what makes the next start fail to bind.

$ErrorActionPreference = "Continue"
. (Join-Path $PSScriptRoot "_common.ps1")

Write-Host ""
Write-Host "  Face Swap Max - stopping" -ForegroundColor Cyan

# Flip the Worker to its Offline page straight away rather than waiting out the
# staleness window, so the public URL does not serve a dead tunnel.
$Secret = Get-AppSecret
if ($Secret) {
    try {
        Invoke-RestMethod -Uri "$PermanentUrl/__offline" -Method Post -TimeoutSec 10 `
            -Headers @{ "x-register-secret" = $Secret } | Out-Null
        Write-Host "    marked offline" -ForegroundColor DarkGray
    } catch { Write-Host "    could not mark offline (Worker unreachable)" -ForegroundColor DarkGray }
}

$servers = @(Get-AppServerProcesses)
$tunnels = @(Get-AppTunnelProcesses)

if ($servers.Count -eq 0 -and $tunnels.Count -eq 0) {
    Write-Host "  nothing running" -ForegroundColor DarkGray
    Write-Host ""
    exit 0
}

$stopped = Stop-AppProcesses

$left = Get-AppListenerPid
if ($left) {
    Write-Host "  port $AppPort is STILL held by pid $left" -ForegroundColor Red
    Write-Host ""
    exit 1
}

Write-Host "  stopped $stopped process(es); port $AppPort is free" -ForegroundColor Green
Write-Host ""

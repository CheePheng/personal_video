# Starts the face swap app and exposes it on a public Cloudflare URL.
#
# Notes that matter:
#  * cloudflared prints its URL on STDERR, so we capture both streams to a file.
#  * cloudflared prints the hostname BEFORE it is actually routable, so we poll
#    /healthz through the tunnel before telling the user it is ready. Otherwise
#    the first click lands on a dead link.

$ErrorActionPreference = "Stop"
$Root      = Split-Path -Parent $PSScriptRoot
$Python    = "C:\fsw\venv\Scripts\python.exe"
$Cloudflared = "C:\Program Files (x86)\cloudflared\cloudflared.exe"
$Port      = 8765
$LogDir    = Join-Path $Root "logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$SrvLog    = Join-Path $LogDir "server.log"
$TunLog    = Join-Path $LogDir "tunnel.log"

function Say($msg, $color = "Gray") { Write-Host $msg -ForegroundColor $color }

if (-not (Test-Path $Python))      { Say "Missing venv at $Python. Run scripts/00_install_runtime.sh first." "Red"; exit 1 }
if (-not (Test-Path $Cloudflared)) { Say "cloudflared not found. Install: winget install Cloudflare.cloudflared" "Red"; exit 1 }

Say ""
Say "  Face Swap" "Cyan"
Say "  ---------" "Cyan"

# --- clear out anything left from a previous run -----------------------------
# Without this, starting again just leaves the OLD server owning port 8765:
# uvicorn fails to bind, and every request keeps hitting the stale process,
# which is still running whatever code was on disk when IT started. Edits to
# the app then appear to do nothing.
Say "  clearing previous run..."

$stale = Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue |
         Where-Object { $_.CommandLine -and $_.CommandLine -like "*uvicorn*app.main*" }
foreach ($p in $stale) {
    try { Stop-Process -Id $p.ProcessId -Force -ErrorAction Stop; Say "    stopped stale process $($p.ProcessId)" "DarkGray" } catch { }
}

# Whoever holds the port wins, so make sure it is actually free before binding.
$holder = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
foreach ($h in $holder) {
    try { Stop-Process -Id $h.OwningProcess -Force -ErrorAction Stop; Say "    freed port $Port" "DarkGray" } catch { }
}
if ($stale -or $holder) { Start-Sleep -Seconds 2 }

# Truncate rather than append: a 4000-line log of old crashes makes the "last
# 20 lines" diagnostic above useless, and hides whether an error is current.
foreach ($f in @($SrvLog, "$SrvLog.err")) { if (Test-Path $f) { Clear-Content $f -Force -ErrorAction SilentlyContinue } }

# --- local server -----------------------------------------------------------
Say "  starting render server..."
$srv = Start-Process -FilePath $Python `
    -ArgumentList "-m","uvicorn","app.main:app","--host","127.0.0.1","--port","$Port","--log-level","warning" `
    -WorkingDirectory $Root -PassThru -NoNewWindow `
    -RedirectStandardOutput $SrvLog -RedirectStandardError "$SrvLog.err"

$ok = $false
foreach ($i in 1..40) {
    Start-Sleep -Milliseconds 500
    try {
        $r = Invoke-WebRequest -Uri "http://127.0.0.1:$Port/healthz" -UseBasicParsing -TimeoutSec 3
        if ($r.StatusCode -eq 200) { $ok = $true; break }
    } catch { }
}
if (-not $ok) {
    Say "  server failed to start. Last lines:" "Red"
    if (Test-Path "$SrvLog.err") { Get-Content "$SrvLog.err" -Tail 20 | ForEach-Object { Say "    $_" "DarkGray" } }
    if ($srv -and -not $srv.HasExited) { Stop-Process -Id $srv.Id -Force }
    exit 1
}
Say "  server up on http://127.0.0.1:$Port" "Green"

# --- public tunnel ----------------------------------------------------------
Say "  opening public tunnel..."
if (Test-Path $TunLog) { Remove-Item $TunLog -Force }
$tun = Start-Process -FilePath $Cloudflared `
    -ArgumentList "tunnel","--url","http://127.0.0.1:$Port","--no-autoupdate" `
    -PassThru -NoNewWindow -RedirectStandardOutput $TunLog -RedirectStandardError "$TunLog.err"

$public = $null
foreach ($i in 1..60) {
    Start-Sleep -Milliseconds 500
    foreach ($f in @($TunLog, "$TunLog.err")) {
        if (Test-Path $f) {
            $m = Select-String -Path $f -Pattern "https://[a-z0-9-]+\.trycloudflare\.com" -AllMatches -ErrorAction SilentlyContinue
            if ($m) { $public = $m.Matches[0].Value; break }
        }
    }
    if ($public) { break }
}

if (-not $public) {
    Say "  could not get a tunnel URL; the app still works locally." "Yellow"
} else {
    # The hostname is printed before DNS/routing settles. Wait for a real 200.
    # DNS for a freshly-minted trycloudflare hostname can take well over a
    # minute to resolve, and cloudflared prints the name before it is routable.
    # Poll until it genuinely answers rather than handing over a dead link.
    Say "  waiting for $public to go live (can take a minute)..."
    $live = $false
    foreach ($i in 1..90) {
        Start-Sleep -Seconds 2
        try {
            $r = Invoke-WebRequest -Uri "$public/healthz" -UseBasicParsing -TimeoutSec 5
            if ($r.StatusCode -eq 200) { $live = $true; break }
        } catch { }
        if ($i % 10 -eq 0) { Say "    still routing... ($($i*2)s)" "DarkGray" }
    }
    if ($live) {
        Say "  tunnel is live" "Green"
    } else {
        Say "  tunnel did not answer in 3 min. Try the link anyway, or restart." "Yellow"
    }
}

# Written by the app on first run; shown here so the link and the password
# always arrive together.
$PwFile = Join-Path $Root "data\.password"
$Pw = if (Test-Path $PwFile) { (Get-Content $PwFile -Raw).Trim() } else { "(see data\.password)" }

# --- tell the Worker where we are ------------------------------------------
# The workers.dev URL is permanent; the tunnel URL behind it is not. Register
# the current one, then heartbeat so the Worker can tell "PC off" from "PC on".
$Permanent  = "https://faceswap.doctorwilddoctorwild.workers.dev"
$SecretFile = Join-Path $Root "data\.register_secret"
$Secret     = if (Test-Path $SecretFile) { (Get-Content $SecretFile -Raw).Trim() } else { $null }
$registered = $false

function Send-Heartbeat {
    param($TunnelUrl)
    if (-not $Secret -or -not $TunnelUrl) { return $false }
    try {
        Invoke-RestMethod -Uri "$Permanent/__register" -Method Post -TimeoutSec 15 `
            -Headers @{ "x-register-secret" = $Secret } `
            -ContentType "application/json" `
            -Body (@{ url = $TunnelUrl } | ConvertTo-Json -Compress) | Out-Null
        return $true
    } catch { return $false }
}

if ($public -and $Secret) {
    Say "  linking permanent URL..."
    $registered = Send-Heartbeat -TunnelUrl $public
    if ($registered) { Say "  permanent URL is live" "Green" }
    else { Say "  could not reach the Worker; use the temporary link below." "Yellow" }
}

Say ""
Say "  ====================================================" "Cyan"
Say "   OPEN THIS LINK" "Cyan"
Say ""
if ($registered) {
    Say "   $Permanent" "White"
    Say ""
    Say "   this link NEVER changes - bookmark it" "DarkGray"
} elseif ($public) {
    Say "   $public" "White"
    Say ""
    Say "   temporary link (changes each restart)" "DarkGray"
} else {
    Say "   http://127.0.0.1:$Port" "White"
}
Say ""
Say "   password:  $Pw" "White"
Say ""
Say "   works from your phone or any device" "DarkGray"
Say "  ====================================================" "Cyan"
Say ""
Say "  Leave this window open. Press Ctrl+C to stop." "DarkGray"
Say ""

try {
    $tick = 0
    while ($true) {
        Start-Sleep -Seconds 2
        if ($srv.HasExited) { Say "  server stopped unexpectedly." "Red"; break }
        # Heartbeat every 60s. If this PC dies or loses power, the Worker sees
        # the gap and starts serving the Offline page on its own.
        $tick += 2
        if ($registered -and $tick -ge 60) {
            $tick = 0
            if (-not (Send-Heartbeat -TunnelUrl $public)) {
                Say "  heartbeat failed (will retry)" "DarkGray"
            }
        }
    }
} finally {
    Say "  shutting down..." "DarkGray"
    # Clean shutdown: flip to Offline immediately rather than waiting out the
    # 3 minute staleness window.
    if ($registered -and $Secret) {
        try {
            Invoke-RestMethod -Uri "$Permanent/__offline" -Method Post -TimeoutSec 10 `
                -Headers @{ "x-register-secret" = $Secret } | Out-Null
            Say "  marked offline" "DarkGray"
        } catch { }
    }
    foreach ($p in @($tun, $srv)) {
        if ($p -and -not $p.HasExited) { try { Stop-Process -Id $p.Id -Force } catch { } }
    }
    # Renders now run inside the server process (app/swapper.py), so stopping
    # the server stops them -- no separate render processes to clean up.
}

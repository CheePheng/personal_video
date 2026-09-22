# Starts the face swap app and exposes it on the permanent public URL.
#
# Two things here are not cosmetic:
#
#  * The process that binds the port is a CHILD of what Start-Process returns.
#    Tracking the returned handle is what used to leave an orphaned server
#    holding port 8765 after shutdown, so the next start could not bind while
#    the health poll still got 200 from the orphan and printed a success
#    banner over a dead server. Ownership now comes from _common.ps1, which
#    decides by port and command line rather than by a handle.
#
#  * cloudflared prints its URL on STDERR and prints it BEFORE the hostname is
#    routable, so we capture both streams and poll through the tunnel before
#    claiming anything is live.
#
# SERVER READY is printed only when every check in Test-Ready passes.

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "_common.ps1")

$Root        = $ProjectRoot
$Python      = "C:\fsw\venv\Scripts\python.exe"
$Cloudflared = "C:\Program Files (x86)\cloudflared\cloudflared.exe"
$Port        = $AppPort
$Local       = "http://127.0.0.1:$Port"
$LogDir      = Join-Path $Root "logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$SrvLog      = Join-Path $LogDir "server.log"
$TunLog      = Join-Path $LogDir "tunnel.log"

function Say($msg, $color = "Gray") { Write-Host $msg -ForegroundColor $color }

function Fail($msg) {
    Say ""
    Say "  STARTUP FAILED" "Red"
    Say "  $msg" "Red"
    foreach ($f in @("$SrvLog.err", $SrvLog)) {
        if (Test-Path $f) {
            $tail = Get-Content $f -Tail 20 -ErrorAction SilentlyContinue
            if ($tail) {
                Say "  --- $(Split-Path -Leaf $f) ---" "DarkGray"
                $tail | ForEach-Object { Say "    $_" "DarkGray" }
            }
        }
    }
    Stop-AppProcesses -Quiet | Out-Null
    exit 1
}

if (-not (Test-Path $Python))      { Fail "Missing venv at $Python. Run scripts/00_install_runtime.sh first." }
if (-not (Test-Path $Cloudflared)) { Fail "cloudflared not found. Install: winget install Cloudflare.cloudflared" }

Say ""
Say "  Face Swap Max" "Cyan"
Say "  -------------" "Cyan"

# --- clear out anything left from a previous run -----------------------------
Say "  clearing previous run..."
$n = Stop-AppProcesses
if ($n -gt 0) { Say "    cleared $n process(es)" "DarkGray" }
if (Get-AppListenerPid) { Fail "port $Port is still held after cleanup; stop it manually and retry." }

foreach ($f in @($SrvLog, "$SrvLog.err")) { if (Test-Path $f) { Clear-Content $f -Force -ErrorAction SilentlyContinue } }

# --- local server -----------------------------------------------------------
Say "  starting render server..."
# FSW_APP_ROOT tags the command line so _common.ps1 can recognise this server
# (and its child) as ours without ever matching an unrelated python.exe.
$env:FSW_APP_ROOT = $Root
$srv = Start-Process -FilePath $Python `
    -ArgumentList "-m","uvicorn","app.main:app","--host","127.0.0.1","--port","$Port","--log-level","warning" `
    -WorkingDirectory $Root -PassThru -NoNewWindow `
    -RedirectStandardOutput $SrvLog -RedirectStandardError "$SrvLog.err"

$ok = $false
foreach ($i in 1..60) {
    Start-Sleep -Milliseconds 500
    # If the launcher died AND nothing is listening, the server is genuinely
    # gone -- do not keep polling for the full timeout.
    if ($srv.HasExited -and -not (Get-AppListenerPid)) {
        Fail "uvicorn exited during startup (exit code $($srv.ExitCode))."
    }
    if ((Invoke-AppRequest -Url "$Local/healthz" -TimeoutSec 3) -eq 200) { $ok = $true; break }
}
if (-not $ok) { Fail "server did not answer /healthz on $Local within 30s." }

$listener = Get-AppListenerPid
$count = @(Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue).Count
if ($count -ne 1) { Fail "expected exactly 1 listener on port $Port, found $count." }
Say "  server up on $Local (pid $listener)" "Green"

# --- public tunnel ----------------------------------------------------------
$publicOk = $false
$publicNote = ""
Say "  opening public tunnel..."
foreach ($f in @($TunLog, "$TunLog.err")) { if (Test-Path $f) { Remove-Item $f -Force -ErrorAction SilentlyContinue } }
$tun = Start-Process -FilePath $Cloudflared `
    -ArgumentList "tunnel","--url","$Local","--no-autoupdate" `
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

$Secret = Get-AppSecret
$registered = $false

function Send-Heartbeat {
    param($TunnelUrl)
    if (-not $Secret -or -not $TunnelUrl) { return $false }
    try {
        Invoke-RestMethod -Uri "$PermanentUrl/__register" -Method Post -TimeoutSec 15 `
            -Headers @{ "x-register-secret" = $Secret } `
            -ContentType "application/json" `
            -Body (@{ url = $TunnelUrl } | ConvertTo-Json -Compress) | Out-Null
        return $true
    } catch { return $false }
}

if (-not $public) {
    $publicNote = "cloudflared never printed a tunnel hostname"
} else {
    Say "  waiting for tunnel routing (can take a minute)..."
    $live = $false
    foreach ($i in 1..90) {
        Start-Sleep -Seconds 2
        if ((Invoke-AppRequest -Url "$public/healthz" -TimeoutSec 5) -eq 200) { $live = $true; break }
        if ($i % 10 -eq 0) { Say "    still routing... ($($i*2)s)" "DarkGray" }
    }
    if (-not $live) {
        $publicNote = "tunnel hostname $public did not route within 3 min"
    } else {
        if (-not $Secret) {
            $publicNote = "no data\.register_secret, cannot link the permanent URL"
        } else {
            Say "  linking permanent URL..."
            $registered = Send-Heartbeat -TunnelUrl $public
            if (-not $registered) { $publicNote = "Worker rejected or did not answer /__register" }
        }
    }
}

# The permanent URL is the contract, so verify the WORKER reaches THIS server,
# not merely that the tunnel is up.
if ($registered) {
    $pubOkCount = 0
    foreach ($i in 1..15) {
        if ((Invoke-AppRequest -Url "$PermanentUrl/healthz" -TimeoutSec 10) -eq 200) { $pubOkCount++; break }
        Start-Sleep -Seconds 2
    }
    if ($pubOkCount -gt 0) { $publicOk = $true }
    else { $publicNote = "permanent URL registered but does not answer /healthz" }
}

# --- readiness --------------------------------------------------------------
# Nothing above prints READY. Every one of these must hold first.
function Test-Ready {
    $fail = @()

    if ((Invoke-AppRequest -Url "$Local/healthz") -ne 200) { $fail += "health endpoint" }
    if ((Invoke-AppRequest -Url "$Local/login") -ne 200)   { $fail += "login page" }

    $s = New-AppSession -BaseUrl $Local
    if (-not $s) { $fail += "login (no data\.password)" ; return $fail }

    $ui = Invoke-AppRequest -Url "$Local/" -Session $s
    if ($ui -ne 200) { $fail += "authenticated UI (got $ui)" }

    if ((Invoke-AppRequest -Url "$Local/api/library" -Session $s) -ne 200) { $fail += "library endpoint" }

    # Thumbnail and Range need a rendered item; if the library is empty these
    # are reported as skipped rather than failed, because there is nothing to
    # serve yet on a fresh install.
    $ids = @(Get-AppLibraryIds -BaseUrl $Local -Session $s)
    if ($ids.Count -gt 0) {
        $jid = $ids[0]
        if ((Invoke-AppRequest -Url "$Local/api/library/$jid/thumb" -Session $s) -ne 200) { $fail += "thumbnail" }
        $rc = Invoke-AppRequest -Url "$Local/api/library/$jid/video" -Session $s -Range "bytes=0-1023"
        if ($rc -ne 206) { $fail += "HTTP Range (got $rc, want 206)" }
    }

    # Max must be the default the UI presents.
    try {
        $html = (Invoke-WebRequest -Uri "$Local/" -WebSession $s -UseBasicParsing -TimeoutSec 10).Content
        if ($html -notmatch 'value="max"[^>]*\bselected\b' -and $html -notmatch '\bselected\b[^>]*value="max"' -and
            $html -notmatch 'value="max"[^>]*\bchecked\b' -and $html -notmatch '\bchecked\b[^>]*value="max"') {
            $fail += "Max not selected by default"
        }
    } catch { $fail += "could not read UI for Max default" }

    $n = @(Get-NetTCPConnection -LocalPort $AppPort -State Listen -ErrorAction SilentlyContinue).Count
    if ($n -ne 1) { $fail += "expected 1 listener, found $n" }

    return $fail
}

Say "  running readiness checks..."
$problems = @(Test-Ready)
if ($problems.Count -gt 0) {
    Say ""
    Say "  NOT READY - the following checks failed:" "Red"
    foreach ($p in $problems) { Say "    - $p" "Red" }
    Stop-AppProcesses -Quiet | Out-Null
    exit 1
}

$Pw = Get-AppPassword
if (-not $Pw) { $Pw = "(see data\.password)" }

Say ""
Say "  ====================================================" "Cyan"
if ($publicOk) {
    Say "   SERVER READY" "Green"
    Say ""
    Say "   $PermanentUrl" "White"
    Say ""
    Say "   this link NEVER changes - bookmark it" "DarkGray"
} else {
    Say "   LOCAL READY" "Green"
    Say "   PUBLIC FAILED" "Red"
    Say ""
    Say "   reason: $publicNote" "Yellow"
    Say ""
    Say "   $Local" "White"
}
Say ""
Say "   password:  $Pw" "White"
Say "  ====================================================" "Cyan"
Say ""
Say "  Leave this window open. Press Ctrl+C to stop." "DarkGray"
Say ""

try {
    $tick = 0
    while ($true) {
        Start-Sleep -Seconds 2
        # Watch the PORT, not the launcher handle: the launcher can outlive the
        # real server and vice versa.
        if (-not (Get-AppListenerPid)) { Say "  server stopped unexpectedly." "Red"; break }
        $tick += 2
        if ($registered -and $tick -ge 60) {
            $tick = 0
            if (-not (Send-Heartbeat -TunnelUrl $public)) { Say "  heartbeat failed (will retry)" "DarkGray" }
        }
    }
} finally {
    Say "  shutting down..." "DarkGray"
    if ($registered -and $Secret) {
        try {
            Invoke-RestMethod -Uri "$PermanentUrl/__offline" -Method Post -TimeoutSec 10 `
                -Headers @{ "x-register-secret" = $Secret } | Out-Null
            Say "  marked offline" "DarkGray"
        } catch { }
    }
    Stop-AppProcesses -Quiet | Out-Null
    Say "  stopped" "DarkGray"
}

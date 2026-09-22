# Shared process identification for start/stop/status.
#
# The whole reliability problem this solves: Start-Process returns a handle to
# a LAUNCHER pid, but on Windows the process that actually binds the port is a
# CHILD of it. Measured, one clean start:
#
#     Start-Process returned pid: 23044
#     listener pid:               44372   (child of 23044)
#
# start.ps1 used to keep $srv = the launcher, poll $srv.HasExited, and on
# shutdown stop only $srv. That orphans the real server: it keeps holding the
# port, so the NEXT start cannot bind, the health poll still gets 200 from the
# orphan, and the banner prints over a server that is not the one just started.
#
# So nothing here trusts a Start-Process handle. Ownership is decided by two
# observable facts instead: the command line names THIS project's app, and the
# process is the one holding THIS project's port. That is also what keeps us
# from killing unrelated python.exe -- a python running someone else's code
# matches neither test.

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$AppPort     = 8765
$PermanentUrl = "https://faceswap.doctorwilddoctorwild.workers.dev"

# Marker placed on the server's command line so it is identifiable beyond
# doubt, even if another project ever runs an app.main on another port.
$AppTag = "FSW_APP_ROOT=$ProjectRoot"

function Get-AppListenerPid {
    $c = Get-NetTCPConnection -LocalPort $AppPort -State Listen -ErrorAction SilentlyContinue
    if ($c) { return @($c)[0].OwningProcess }
    return $null
}

function Get-AppServerProcesses {
    # A process is ours if it is a python running uvicorn on app.main AND it
    # carries our root marker, OR it is the process currently holding our port.
    # The second clause catches the orphaned child, whose command line is
    # inherited from the launcher and so carries the marker too.
    $out = @{}
    $procs = Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue |
             Where-Object { $_.CommandLine -and $_.CommandLine -like "*uvicorn*app.main*" }
    foreach ($p in $procs) {
        # Only ours: must reference this project root, either via the explicit
        # marker or by running out of this directory.
        if ($p.CommandLine -like "*$AppTag*" -or $p.CommandLine -like "*$ProjectRoot*") {
            $out[$p.ProcessId] = $p
            continue
        }
        # Legacy processes started before the marker existed: claim them only
        # if they hold OUR port. Anything else is left strictly alone.
        $lp = Get-AppListenerPid
        if ($lp -and $p.ProcessId -eq $lp) { $out[$p.ProcessId] = $p }
    }
    # Include the port holder and its children even if the name check missed.
    $lp = Get-AppListenerPid
    if ($lp -and -not $out.ContainsKey($lp)) {
        $h = Get-CimInstance Win32_Process -Filter "ProcessId=$lp" -ErrorAction SilentlyContinue
        if ($h -and $h.Name -eq "python.exe") { $out[$lp] = $h }
    }
    # Pull in children of anything already claimed -- that is the real server.
    $all = Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue
    $added = $true
    while ($added) {
        $added = $false
        foreach ($p in $all) {
            if ($out.ContainsKey($p.ProcessId)) { continue }
            if ($out.ContainsKey($p.ParentProcessId)) { $out[$p.ProcessId] = $p; $added = $true }
        }
    }
    return $out.Values
}

function Get-AppTunnelProcesses {
    # Only a cloudflared pointed at OUR port. A cloudflared serving anything
    # else belongs to someone else and is never touched.
    Get-CimInstance Win32_Process -Filter "Name='cloudflared.exe'" -ErrorAction SilentlyContinue |
        Where-Object { $_.CommandLine -and $_.CommandLine -like "*127.0.0.1:$AppPort*" }
}

function Stop-AppProcesses {
    param([switch]$Quiet)
    $stopped = 0
    # Children first, so a parent cannot respawn or hold the port after.
    $servers = @(Get-AppServerProcesses) | Sort-Object -Property ParentProcessId -Descending
    foreach ($p in $servers) {
        try {
            Stop-Process -Id $p.ProcessId -Force -ErrorAction Stop
            $stopped++
            if (-not $Quiet) { Write-Host ("    stopped server pid " + $p.ProcessId) -ForegroundColor DarkGray }
        } catch { }
    }
    foreach ($p in @(Get-AppTunnelProcesses)) {
        try {
            Stop-Process -Id $p.ProcessId -Force -ErrorAction Stop
            $stopped++
            if (-not $Quiet) { Write-Host ("    stopped tunnel pid " + $p.ProcessId) -ForegroundColor DarkGray }
        } catch { }
    }
    # Wait for the port to actually clear; a killed process releases it
    # asynchronously and the next bind fails if we return too early.
    foreach ($i in 1..40) {
        if (-not (Get-AppListenerPid)) { break }
        Start-Sleep -Milliseconds 250
    }
    return $stopped
}

function Get-AppPassword {
    $f = Join-Path $ProjectRoot "data\.password"
    if (Test-Path $f) { return (Get-Content $f -Raw).Trim() }
    return $null
}

function Get-AppSecret {
    $f = Join-Path $ProjectRoot "data\.register_secret"
    if (Test-Path $f) { return (Get-Content $f -Raw).Trim() }
    return $null
}

function Invoke-AppRequest {
    # Returns the status code, or 0 when the request could not be made at all.
    #
    # Range goes through curl.exe rather than Invoke-WebRequest: Range is a
    # restricted .NET header, so passing it via -Headers throws ("must be
    # modified using the appropriate property or method") -- and that throw
    # also leaves the WebSession unusable for later calls.
    param([string]$Url, [int]$TimeoutSec = 8, $Session = $null, [string]$Range = $null)
    if ($Range) {
        $cargs = @("-s","-o","NUL","-w","%{http_code}","--max-time","$TimeoutSec","-H","Range: $Range")
        if ($Session) {
            $c = $Session.Cookies.GetCookies([Uri]$Url)
            foreach ($ck in $c) { $cargs += @("-b", ($ck.Name + "=" + $ck.Value)) }
        }
        $cargs += $Url
        $code = & curl.exe @cargs 2>$null
        if ($code -match '^\d+$') { return [int]$code }
        return 0
    }
    try {
        $p = @{ Uri = $Url; UseBasicParsing = $true; TimeoutSec = $TimeoutSec;
                ErrorAction = "Stop" }
        if ($Session) { $p["WebSession"] = $Session }
        $r = Invoke-WebRequest @p
        return [int]$r.StatusCode
    } catch {
        $resp = $_.Exception.Response
        if ($resp -and $resp.StatusCode) { return [int]$resp.StatusCode }
        return 0
    }
}

function Get-AppLibraryIds {
    # /api/library returns a bare JSON array.
    #
    # Windows PowerShell 5.1 returns such an array from ConvertFrom-Json as a
    # SINGLE Object[] element, so @(...) gives Count=1 and [0] is the whole
    # array -- reading .id off it then yields every id joined into one string,
    # which produces a 404 on the "first" thumbnail. Piping through
    # ForEach-Object enumerates the rows properly.
    param([string]$BaseUrl, $Session)
    try {
        $raw = Invoke-WebRequest -Uri "$BaseUrl/api/library" -WebSession $Session `
                 -UseBasicParsing -TimeoutSec 10 -ErrorAction Stop
        $arr = $raw.Content | ConvertFrom-Json | ForEach-Object { $_ }
        if ($null -eq $arr) { return @() }
        return @($arr | ForEach-Object { $_.id })
    } catch { return @() }
}

function New-AppSession {
    # Logs in and returns a session carrying the auth cookie, or $null.
    #
    # /login answers 303 to "/" and sets the cookie on that response. Under
    # -MaximumRedirection 0 PowerShell treats the 303 as a terminating error,
    # so the session MUST be created outside the try -- otherwise the catch
    # returns an out-of-scope variable and every authenticated check silently
    # runs unauthenticated. Letting the redirect be followed is simpler and
    # leaves the cookie in the session either way.
    param([string]$BaseUrl)
    $pw = Get-AppPassword
    if (-not $pw) { return $null }
    $s = New-Object Microsoft.PowerShell.Commands.WebRequestSession
    try {
        Invoke-WebRequest -Uri "$BaseUrl/login" -Method Post -Body @{ password = $pw } `
            -WebSession $s -UseBasicParsing -TimeoutSec 15 -ErrorAction Stop | Out-Null
    } catch { }
    # Confirm the cookie actually works rather than assuming the POST took.
    $code = Invoke-AppRequest -Url "$BaseUrl/" -Session $s
    if ($code -ne 200) { return $null }
    return $s
}

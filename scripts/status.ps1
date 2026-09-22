# Reports whether the app is actually usable right now.
#
# Every line is measured, never assumed: the server is "RUNNING" because
# something is listening on our port and answering, not because a pid file
# exists. Read-only -- this never starts or stops anything.
#
# The password is deliberately not printed.

$ErrorActionPreference = "Continue"
. (Join-Path $PSScriptRoot "_common.ps1")

$Local = "http://127.0.0.1:$AppPort"

function Line($label, $value, $color = "Gray") {
    Write-Host ("  {0,-14}" -f ($label + ":")) -NoNewline -ForegroundColor DarkGray
    Write-Host $value -ForegroundColor $color
}

Write-Host ""
Write-Host "  Face Swap Max" -ForegroundColor Cyan
Write-Host "  -------------" -ForegroundColor Cyan

$listener = Get-AppListenerPid
$listeners = @(Get-NetTCPConnection -LocalPort $AppPort -State Listen -ErrorAction SilentlyContinue).Count

if (-not $listener) {
    Line "Server" "STOPPED" "Red"
    Line "Listeners" "0" "Red"
    Write-Host ""
    Write-Host "  start it with:  powershell -File scripts\start.ps1" -ForegroundColor DarkGray
    Write-Host ""
    exit 1
}

Line "Server" "RUNNING" "Green"
Line "PID" $listener
Line "Port" $AppPort

$health = Invoke-AppRequest -Url "$Local/healthz"
Line "Local" $health $(if ($health -eq 200) { "Green" } else { "Red" })

$loginCode = Invoke-AppRequest -Url "$Local/login"
Line "Login" $(if ($loginCode -eq 200) { "OK" } else { "FAILED ($loginCode)" }) `
     $(if ($loginCode -eq 200) { "Green" } else { "Red" })

$pub = Invoke-AppRequest -Url "$PermanentUrl/healthz" -TimeoutSec 20
Line "Public" $(if ($pub -eq 200) { "200" } else { "$pub  (permanent URL not reaching this server)" }) `
     $(if ($pub -eq 200) { "Green" } else { "Yellow" })

$s = New-AppSession -BaseUrl $Local
if ($s) {
    $ui = Invoke-AppRequest -Url "$Local/" -Session $s
    Line "UI" $ui $(if ($ui -eq 200) { "Green" } else { "Red" })

    # Max default, read from the markup the user actually gets.
    try {
        $html = (Invoke-WebRequest -Uri "$Local/" -WebSession $s -UseBasicParsing -TimeoutSec 10).Content
        $maxOk = ($html -match 'value="max"[^>]*\bselected\b') -or ($html -match '\bselected\b[^>]*value="max"')
        Line "Max" $(if ($maxOk) { "selected" } else { "NOT DEFAULT" }) $(if ($maxOk) { "Green" } else { "Red" })
    } catch { Line "Max" "unknown" "Yellow" }

    $libCode = Invoke-AppRequest -Url "$Local/api/library" -Session $s
    Line "Library" $(if ($libCode -eq 200) { "OK" } else { "FAILED ($libCode)" }) `
         $(if ($libCode -eq 200) { "Green" } else { "Red" })

    try {
        $items = @(Get-AppLibraryIds -BaseUrl $Local -Session $s)
        Line "Renders" $items.Count
        if ($items.Count -gt 0) {
            $jid = $items[0]
            $t = Invoke-AppRequest -Url "$Local/api/library/$jid/thumb" -Session $s
            Line "Thumbnail" $(if ($t -eq 200) { "OK" } else { "FAILED ($t)" }) `
                 $(if ($t -eq 200) { "Green" } else { "Red" })
            $rc = Invoke-AppRequest -Url "$Local/api/library/$jid/video" -Session $s -Range "bytes=0-1023"
            Line "HTTP Range" $rc $(if ($rc -eq 206) { "Green" } else { "Red" })
        } else {
            Line "Thumbnail" "no renders yet" "DarkGray"
            Line "HTTP Range" "no renders yet" "DarkGray"
        }
    } catch { Line "Library" "unreadable" "Red" }

    # Active jobs now, plus whether the LATEST job failed. An all-time error
    # count is not a health signal -- old failures from since-fixed bugs stay
    # in the database forever and would make a healthy server look broken.
    try {
        # Parse the raw JSON rather than letting Invoke-RestMethod unroll the
        # array -- the unrolled form makes property access return EVERY row's
        # value joined together instead of one row's.
        $raw = Invoke-WebRequest -Uri "$Local/api/jobs" -WebSession $s `
                 -UseBasicParsing -TimeoutSec 10 -ErrorAction Stop
        # Windows PowerShell 5.1 hands back a JSON array as ONE Object[]
        # element, so @(...) around it gives Count=1 and [0] is the whole
        # array. Piping through ForEach-Object enumerates it properly.
        $jobs = @($raw.Content | ConvertFrom-Json | ForEach-Object { $_ })
        $active = @($jobs | Where-Object { $_.status -eq "running" -or $_.status -eq "queued" })
        Line "Active jobs" $active.Count
        $sorted = @($jobs | Sort-Object -Property created_at -Descending)
        if ($sorted.Count -gt 0) {
            $st = [string]$sorted[0].status
            Line "Last job" $st $(if ($st -eq "done") { "Green" }
                                  elseif ($st -eq "error") { "Red" }
                                  else { "Gray" })
        }
    } catch { }
} else {
    Line "UI" "cannot log in (no data\.password)" "Yellow"
}

Line "Listeners" $listeners $(if ($listeners -eq 1) { "Green" } else { "Red" })

$tun = @(Get-AppTunnelProcesses)
Line "Tunnel" $(if ($tun.Count -eq 1) { "1 (pid " + $tun[0].ProcessId + ")" } elseif ($tun.Count -eq 0) { "none" } else { $tun.Count.ToString() + " (expected 1)" }) `
     $(if ($tun.Count -eq 1) { "Green" } else { "Yellow" })

# GPU, straight from the runtime the app uses.
try {
    $gpu = & "C:\fsw\venv\Scripts\python.exe" -c "import torch; print(('CUDA / ' + torch.cuda.get_device_name(0)) if torch.cuda.is_available() else 'CPU ONLY')" 2>$null
    if ($gpu) { Line "GPU" $gpu $(if ($gpu -like "CUDA*") { "Green" } else { "Red" }) }
} catch { }

Write-Host ""

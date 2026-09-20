#!/usr/bin/env bash
# Empirically finds the max request body size through a Cloudflare Quick Tunnel.
# This decides whether we need chunked uploads. Docs are ambiguous; measure it.
set -u
CF="/c/Program Files (x86)/cloudflared/cloudflared.exe"
LOG=/c/fsw/tunnel.log
SRVLOG=/c/fsw/probe_server.log
rm -f "$LOG" "$SRVLOG"

echo "=== starting probe server ==="
python "C:/Users/PC/Downloads/myproject_idea/personal_video/scripts/probe_server.py" > "$SRVLOG" 2>&1 &
SRV=$!
sleep 3
curl -s --max-time 5 http://127.0.0.1:8787/ && echo "local server OK" || echo "local server FAILED"

echo "=== starting quick tunnel (no account needed) ==="
"$CF" tunnel --url http://127.0.0.1:8787 --no-autoupdate > "$LOG" 2>&1 &
TUN=$!

URL=""
for i in $(seq 1 45); do
  URL=$(grep -ohE 'https://[a-z0-9-]+\.trycloudflare\.com' "$LOG" 2>/dev/null | head -1)
  [ -n "$URL" ] && break
  sleep 2
done

if [ -z "$URL" ]; then
  echo "TUNNEL_URL=NONE"; echo "--- tunnel log ---"; tail -40 "$LOG"
  kill $SRV $TUN 2>/dev/null; exit 1
fi
echo "TUNNEL_URL=$URL"
sleep 5
echo "=== reachability ==="
curl -s --max-time 30 "$URL/" || echo "tunnel GET failed"

echo "=== upload size sweep ==="
mkdir -p /c/fsw/testdata
for MB in 1 50 95 99 105 150 300 600; do
  F="/c/fsw/testdata/t${MB}.bin"
  [ -f "$F" ] || head -c $((MB*1024*1024)) /dev/urandom > "$F"
  START=$(date +%s)
  RES=$(curl -s --max-time 900 -w "|HTTP=%{http_code}|T=%{time_total}s|SPD=%{speed_upload}" \
        -X POST --data-binary "@$F" -H "Content-Type: application/octet-stream" "$URL/upload" 2>&1 | tr -d '\r')
  END=$(date +%s)
  echo "SIZE=${MB}MB  ${RES}  wall=$((END-START))s"
done

echo "=== server-side view ==="
tail -20 "$SRVLOG"
kill $SRV $TUN 2>/dev/null
echo "=== sweep complete ==="

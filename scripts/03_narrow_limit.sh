#!/usr/bin/env bash
set -u
CF="/c/Program Files (x86)/cloudflared/cloudflared.exe"
LOG=/c/fsw/tunnel2.log; rm -f "$LOG"
python "C:/Users/PC/Downloads/myproject_idea/personal_video/scripts/probe_server.py" > /c/fsw/probe2.log 2>&1 &
SRV=$!; sleep 3
"$CF" tunnel --url http://127.0.0.1:8787 --no-autoupdate > "$LOG" 2>&1 &
TUN=$!
URL=""
for i in $(seq 1 45); do
  URL=$(grep -ohE 'https://[a-z0-9-]+\.trycloudflare\.com' "$LOG" 2>/dev/null | head -1)
  [ -n "$URL" ] && break; sleep 2
done
echo "URL=$URL"; sleep 5
for MB in 350 400 450 500 512 550; do
  F="/c/fsw/testdata/t${MB}.bin"; [ -f "$F" ] || head -c $((MB*1024*1024)) /dev/urandom > "$F"
  CODE=$(curl -s -o /dev/null -w "%{http_code}|%{time_total}s" --max-time 600 -X POST \
        --data-binary "@$F" -H "Content-Type: application/octet-stream" "$URL/upload" 2>&1)
  echo "NARROW ${MB}MB -> $CODE"
done
kill $SRV $TUN 2>/dev/null; echo "NARROW DONE"

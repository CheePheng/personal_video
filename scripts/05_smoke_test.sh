#!/usr/bin/env bash
# Boots the API and exercises the real endpoints end to end (no GPU render).
set -u
cd "C:/Users/PC/Downloads/myproject_idea/personal_video" || exit 1
PY=/c/fsw/venv/Scripts/python.exe
"$PY" -m uvicorn app.main:app --host 127.0.0.1 --port 8765 --log-level warning > /c/fsw/smoke_srv.log 2>&1 &
SRV=$!
for i in $(seq 1 40); do
  curl -s --max-time 2 http://127.0.0.1:8765/healthz > /dev/null 2>&1 && break
  sleep 0.5
done
echo "=== /healthz ==="; curl -s --max-time 5 http://127.0.0.1:8765/healthz; echo
echo "=== / (bytes) ==="; curl -s --max-time 5 http://127.0.0.1:8765/ | wc -c
echo "=== /static/app.js (bytes) ==="; curl -s --max-time 5 http://127.0.0.1:8765/static/app.js | wc -c
echo "=== chunked upload round-trip (12MB -> 2 chunks) ==="
head -c $((12*1024*1024)) /dev/urandom > /c/fsw/up_test.bin
UID_=$(curl -s -X POST -H 'Content-Type: application/json' \
  -d '{"filename":"up_test.bin","size":12582912}' http://127.0.0.1:8765/api/upload/init \
  | python -c "import sys,json;print(json.load(sys.stdin)['upload_id'])")
echo "upload_id=$UID_"
split -b 8388608 /c/fsw/up_test.bin /c/fsw/chunk_
i=0
for f in /c/fsw/chunk_*; do
  curl -s -X POST --data-binary "@$f" -H 'Content-Type: application/octet-stream' \
    "http://127.0.0.1:8765/api/upload/chunk/$UID_/$i" ; echo " <- chunk $i"
  i=$((i+1))
done
curl -s -X POST -H 'Content-Type: application/json' \
  -d "{\"upload_id\":\"$UID_\",\"chunks\":$i}" http://127.0.0.1:8765/api/upload/complete; echo
echo "=== integrity ==="
ASM=$(ls -S /c/fsw/../Users/PC/Downloads/myproject_idea/personal_video/data/uploads/*up_test.bin 2>/dev/null | head -1)
if [ -n "$ASM" ]; then
  echo "orig: $(md5sum /c/fsw/up_test.bin | cut -d' ' -f1)"
  echo "asm : $(md5sum "$ASM" | cut -d' ' -f1)"
fi
echo "=== /api/jobs list ==="; curl -s --max-time 5 http://127.0.0.1:8765/api/jobs; echo
rm -f /c/fsw/chunk_*
kill $SRV 2>/dev/null
echo "SMOKE DONE"

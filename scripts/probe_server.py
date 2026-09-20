"""Minimal upload sink used ONLY to measure the real max request body size
through a Cloudflare Quick Tunnel. Reads the body in chunks, discards it,
and reports how many bytes actually arrived."""
import http.server, socketserver, sys

PORT = 8787

class H(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        got = 0
        try:
            while got < n:
                c = self.rfile.read(min(1 << 20, n - got))
                if not c:
                    break
                got += len(c)
        except Exception as e:
            sys.stderr.write(f"read error after {got} bytes: {e!r}\n")
        body = f"received={got} declared={n}\n".encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/plain")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
        sys.stderr.write(f"POST ok received={got} declared={n}\n")
    def do_GET(self):
        b = b"probe-ok\n"
        self.send_response(200)
        self.send_header("Content-Type", "text/plain")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)
    def log_message(self, *a): pass

class S(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True

print(f"probe server on {PORT}", flush=True)
S(("127.0.0.1", PORT), H).serve_forever()

import http.server, functools, socketserver, json, os, sys, traceback
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
D=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
class H(http.server.SimpleHTTPRequestHandler):
    def do_POST(self):
        if self.path.startswith("/solve"):
            n = int(self.headers.get("Content-Length", 0))
            try:
                req = json.loads(self.rfile.read(n)) if n else {}
                import solve_api
                out = solve_api.solve(req.get("view", "red_station"),
                                      pts=req.get("points"), lines=req.get("lines"),
                                      lens_from=req.get("lens_from"))
                body = json.dumps(out).encode()
            except Exception as e:
                body = json.dumps(dict(error=f"{e}", tb=traceback.format_exc()[-800:])).encode()
            self.send_response(200); self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers(); self.wfile.write(body); return

        dest = {"/save": "out/labels.json", "/trace": "out/trace.json",
                "/points": "out/points_multi.json",
                "/pairs": "out/pairs_xview.json",
                "/lines": "out/lines_multi.json"}.get(self.path)
        if dest is None:
            self.send_error(404); return
        n=int(self.headers.get("Content-Length",0))
        body=self.rfile.read(n)
        try: json.loads(body)                      # reject anything not valid JSON
        except Exception as e:
            self.send_error(400, f"bad json: {e}"); return
        # Never overwrite annotation work without keeping the previous copy.
        # Hand-marked points are expensive and a bad save should not be able to
        # destroy them.
        full = os.path.join(D, dest)
        if os.path.exists(full):
            import shutil, time
            bdir = os.path.join(D, "out", "backup"); os.makedirs(bdir, exist_ok=True)
            stem = os.path.basename(dest)[:-5]
            shutil.copy2(full, os.path.join(
                bdir, f"{stem}-{time.strftime('%Y%m%d-%H%M%S')}.json"))
        with open(full, "wb") as f: f.write(body)
        self.send_response(200); self.send_header("Content-Type","text/plain")
        self.send_header("Access-Control-Allow-Origin","*")
        self.end_headers(); self.wfile.write(f"saved {n} bytes".encode())
    def end_headers(self):
        self.send_header("Cache-Control","no-store"); super().end_headers()
Handler=functools.partial(H, directory=D)
socketserver.TCPServer.allow_reuse_address=True
with socketserver.TCPServer(("127.0.0.1",8765), Handler) as s:
    print("serving", D, "on 8765 (POST /save -> out/labels.json, /trace, /points)", flush=True)
    s.serve_forever()

"""
local_server.py – run locally for testing before Vercel deploy
Usage: python local_server.py
Then open http://localhost:8000
"""
import http.server, socketserver, threading, webbrowser, os, json, sys
sys.path.insert(0, os.path.dirname(__file__))

from api.predict import predict_classification, predict_regression, predict_clustering
from urllib.parse import urlparse, parse_qs

PORT = 8000

PERMISSIVE_HEADERS = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": "GET, POST, OPTIONS, PUT, DELETE",
    "Access-Control-Allow-Headers": "*",
    "Content-Security-Policy": "default-src * 'unsafe-inline' 'unsafe-eval' data: blob:;",
    "X-Content-Type-Options": "nosniff",
    "Permissions-Policy": "interest-cohort=()",
}

class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=os.path.join(os.path.dirname(__file__), "public"), **kwargs)

    def send_permissive_headers(self):
        for k, v in PERMISSIVE_HEADERS.items():
            self.send_header(k, v)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_permissive_headers()
        self.end_headers()

    def do_GET(self):
        super().do_GET()

    def end_headers(self):
        self.send_permissive_headers()
        super().end_headers()

    def do_POST(self):
        if self.path.startswith("/api/predict"):
            qs    = parse_qs(urlparse(self.path).query)
            track = qs.get("track", ["classification"])[0]
            length = int(self.headers.get("Content-Length", 0))
            body  = json.loads(self.rfile.read(length) or b"{}")
            try:
                if track == "classification":
                    result = predict_classification(body)
                elif track == "regression":
                    result = predict_regression(body)
                elif track == "clustering":
                    result = predict_clustering(body)
                else:
                    result = {"error": "Unknown track"}
                code = 200
            except Exception as e:
                result = {"error": str(e)}
                code = 500

            out = json.dumps(result).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)
        else:
            self.send_error(404)

    def log_message(self, fmt, *args):
        print(f"[{self.address_string()}] {fmt % args}")

if __name__ == "__main__":
    print(f"Starting local dev server at http://localhost:{PORT}")
    print("Press Ctrl+C to stop.")
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        webbrowser.open(f"http://localhost:{PORT}")
        httpd.serve_forever()

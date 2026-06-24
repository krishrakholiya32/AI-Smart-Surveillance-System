"""
Webcam MJPEG server — run this on your host machine (NOT inside Docker).
Docker containers can then access your webcam via:
  http://host.docker.internal:8765/video

Usage:
    python scripts/webcam_server.py          # webcam index 0
    python scripts/webcam_server.py 1        # webcam index 1
    python scripts/webcam_server.py --port 8765
"""
import sys
import time
import argparse
import threading
from socketserver import ThreadingMixIn
from http.server import BaseHTTPRequestHandler, HTTPServer

try:
    import cv2
except ImportError:
    print("OpenCV not found. Install it: pip install opencv-python")
    sys.exit(1)

frame_lock = threading.Lock()
latest_frame = None
cap = None


def capture_loop(index: int):
    global latest_frame, cap
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        print(f"ERROR: Could not open webcam index {index}")
        sys.exit(1)
    print(f"Webcam {index} opened OK")
    while True:
        ret, frame = cap.read()
        if ret:
            _, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            with frame_lock:
                latest_frame = jpg.tobytes()
        else:
            time.sleep(0.01)


class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


class MJPEGHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass  # suppress per-request logs

    def do_GET(self):
        if self.path == "/video":
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            try:
                while True:
                    with frame_lock:
                        frame = latest_frame
                    if frame:
                        header = (
                            b"--frame\r\n"
                            b"Content-Type: image/jpeg\r\n"
                            + f"Content-Length: {len(frame)}\r\n\r\n".encode()
                        )
                        self.wfile.write(header)
                        self.wfile.write(frame)
                        self.wfile.write(b"\r\n")
                        self.wfile.flush()
                    time.sleep(0.033)  # ~30 fps
            except (BrokenPipeError, ConnectionResetError):
                pass
        elif self.path == "/health":
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"ok")
        else:
            self.send_response(404)
            self.end_headers()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("index", nargs="?", type=int, default=0, help="Webcam index (default: 0)")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    t = threading.Thread(target=capture_loop, args=(args.index,), daemon=True)
    t.start()

    # Wait for first frame
    print("Waiting for first frame...", end="", flush=True)
    while latest_frame is None:
        time.sleep(0.1)
    print(" ready!")

    server = ThreadingHTTPServer(("0.0.0.0", args.port), MJPEGHandler)
    print(f"\nWebcam MJPEG server running on port {args.port}")
    print(f"  Local:  http://localhost:{args.port}/video")
    print(f"  Docker: http://host.docker.internal:{args.port}/video")
    print(f"\nIn the app, add a camera with source:")
    print(f"  http://host.docker.internal:{args.port}/video")
    print("\nPress Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")

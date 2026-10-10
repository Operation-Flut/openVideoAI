"""
openVideoAI - Main entry point
Run the FastAPI server with model management API and UI
Supports both browser mode and native desktop window mode
"""
import sys
import os
import socket
import threading
import time
import argparse

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Make all console output (including third-party progress prints such as
# RealESRGAN's "Tile x/y") appear immediately, even when stdout is redirected.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(line_buffering=True)
    except (AttributeError, ValueError, OSError):
        pass


def find_free_port(host: str, preferred_port: int, tries: int = 20) -> int:
    """Return the first free port at/above preferred_port, or None if none is free."""
    for port in range(preferred_port, preferred_port + tries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind((host, port))
                return port
            except OSError:
                continue
    return None


def run_server(host="127.0.0.1", port=8000, server_holder=None):
    """Run the FastAPI server.

    The running uvicorn.Server instance is stored in server_holder (a dict) so
    desktop/browser modes can stop it cleanly when the window closes.
    """
    import uvicorn
    from src.api import app

    config = uvicorn.Config(app, host=host, port=port, log_level="info")
    server = uvicorn.Server(config)
    if server_holder is not None:
        server_holder["server"] = server
    server.run()


def stop_server(server_holder, timeout=5.0):
    """Ask the uvicorn server to stop and join its thread."""
    server = server_holder.get("server")
    if server is not None:
        server.should_exit = True
    thread = server_holder.get("thread")
    if thread is not None and thread.is_alive():
        thread.join(timeout)


def _request_processing_shutdown():
    """Notify the backend so in-flight processing tasks abort cooperatively."""
    try:
        from src import api as api_module
        api_module.request_processing_shutdown()
        return api_module
    except Exception as e:
        print(f"  (could not notify processing workers: {e})", flush=True)
        return None


def run_desktop_window(host="127.0.0.1", port=8000, width=1280, height=800):
    """Run the app in a native desktop window using pywebview"""
    try:
        import webview
    except ImportError:
        print("pywebview not installed. Install with: pip install pywebview")
        print("Falling back to browser mode...")
        run_server(host, port)
        return

    # Start server in a daemon background thread
    server_holder = {"thread": None}
    server_thread = threading.Thread(
        target=run_server,
        args=(host, port, server_holder),
        daemon=True
    )
    server_holder["thread"] = server_thread
    server_thread.start()

    # Wait for server to start
    print(f"Starting server on http://{host}:{port}...", flush=True)
    time.sleep(2)

    # Create native window
    window = webview.create_window(
        "openVideoAI",
        f"http://{host}:{port}",
        width=width,
        height=height,
        min_size=(1024, 700),
        background_color="#121212"
    )

    # Start the webview event loop (blocks until window closed)
    webview.start(debug=False)

    # The window is gone: abort any in-flight work, stop the server and exit.
    # Without this the process kept running in the background, finished the
    # video, flooded the console with progress and held the port for the next
    # launch (-> "address already in use").
    print("Application closed. Shutting down...", flush=True)
    api_module = _request_processing_shutdown()

    stop_server(server_holder, timeout=1.0)

    # Give an in-flight worker a short grace period to abort and release files
    if api_module is not None:
        try:
            deadline = time.time() + 30
            while time.time() < deadline and api_module.has_active_processing():
                time.sleep(0.2)
        except Exception:
            pass

    stop_server(server_holder, timeout=3.0)
    print("Server stopped.", flush=True)

    # Guarantee the process exits. pywebview may leave non-daemon threads behind
    # which otherwise prevent the interpreter from shutting down.
    os._exit(0)


def run_browser_mode(host="127.0.0.1", port=8000):
    """Run server and open in default browser"""
    import webbrowser

    # Start server in a daemon background thread
    server_holder = {"thread": None}
    server_thread = threading.Thread(
        target=run_server,
        args=(host, port, server_holder),
        daemon=True
    )
    server_holder["thread"] = server_thread
    server_thread.start()

    # Wait for server to start
    print(f"Starting server on http://{host}:{port}...", flush=True)
    time.sleep(2)

    # Open browser
    url = f"http://{host}:{port}"
    print(f"Opening {url} in browser...", flush=True)
    webbrowser.open(url)

    # Keep main thread alive until Ctrl+C
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down...", flush=True)
        _request_processing_shutdown()
        stop_server(server_holder)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="openVideoAI - Local AI Video Enhancement")
    parser.add_argument("--mode", choices=["window", "browser", "server"], default="window",
                        help="Run mode: window (native desktop), browser (open in browser), server (API only)")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000,
                        help="Preferred port (automatically picks the next free port if busy)")
    parser.add_argument("--width", type=int, default=1280, help="Window width (window mode)")
    parser.add_argument("--height", type=int, default=800, help="Window height (window mode)")

    args = parser.parse_args()

    # Pick a free port automatically so a leftover instance cannot brick startup
    effective_port = find_free_port(args.host, args.port)
    if effective_port is None:
        print(f"ERROR: No free port available starting from {args.port}. "
              f"Close other openVideoAI instances and retry.", flush=True)
        sys.exit(1)

    print("=" * 50)
    print("openVideoAI - Local AI Video Enhancement")
    print("=" * 50)
    print(f"Mode: {args.mode}")
    print(f"Host: {args.host}")
    if effective_port != args.port:
        print(f"Port: {args.port} (in use — using {effective_port})")
    else:
        print(f"Port: {effective_port}")
    print(f"URL: http://{args.host}:{effective_port}")

    if args.mode == "window":
        run_desktop_window(args.host, effective_port, args.width, args.height)
    elif args.mode == "browser":
        run_browser_mode(args.host, effective_port)
    else:
        run_server(args.host, effective_port)
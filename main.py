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
import logging

# Add src to path (needed when running from source; no-op when frozen)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# ──────────────────────────────────────────────────────────────────────────────
# Logging setup
# In the distributed (frozen) build stdout/stderr go nowhere visible, so we
# always write to a rotating log file inside %APPDATA%/openVideoAI/logs/.
# In dev mode we just leave the normal console output in place.
# ──────────────────────────────────────────────────────────────────────────────
def _setup_logging() -> None:
    from src.core.paths import is_frozen, log_file

    handlers: list[logging.Handler] = []

    if is_frozen():
        # Distributed build: file only (console is invisible anyway)
        try:
            from logging.handlers import RotatingFileHandler
            lf = log_file()
            fh = RotatingFileHandler(
                lf, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
            )
            fh.setFormatter(logging.Formatter(
                "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            ))
            handlers.append(fh)

            # Also redirect raw stdout/stderr to the same file so that
            # third-party progress prints end up in the log.
            _logfile_obj = open(lf, "a", encoding="utf-8", buffering=1)
            sys.stdout = _logfile_obj
            sys.stderr = _logfile_obj
        except Exception as exc:
            # If we can't open the log file, fall back silently
            print(f"Warning: could not set up log file: {exc}", flush=True)
            handlers.append(logging.StreamHandler(sys.stdout))
    else:
        # Dev mode: console only, immediately flushed
        for _stream in (sys.stdout, sys.stderr):
            try:
                _stream.reconfigure(line_buffering=True)
            except (AttributeError, ValueError, OSError):
                pass
        handlers.append(logging.StreamHandler(sys.stdout))

    logging.basicConfig(
        level=logging.INFO,
        handlers=handlers,
    )


_setup_logging()

log = logging.getLogger("openVideoAI")


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
        log.warning("Could not notify processing workers: %s", e)
        return None


def run_desktop_window(host="127.0.0.1", port=8000, width=1280, height=800):
    """Run the app in a native desktop window using pywebview"""
    try:
        import webview
    except ImportError:
        log.error("pywebview not installed. Install with: pip install pywebview")
        log.info("Falling back to browser mode...")
        run_server(host, port)
        return

    # Resolve the window icon path
    from src.core.paths import icon_path
    _icon = str(icon_path()) if icon_path().exists() else None

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
    log.info("Starting server on http://%s:%s...", host, port)
    time.sleep(2)

    # Create native window
    window = webview.create_window(
        "openVideoAI",
        f"http://{host}:{port}",
        width=width,
        height=height,
        min_size=(1024, 700),
        background_color="#121212",
    )

    # pywebview ≥ 4 accepts an `icon` kwarg on start() (not create_window)
    start_kwargs: dict = {"debug": False}
    if _icon:
        start_kwargs["icon"] = _icon

    # Start the webview event loop (blocks until window closed)
    webview.start(**start_kwargs)

    # The window is gone: abort any in-flight work, stop the server and exit.
    log.info("Application closed. Shutting down...")
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
    log.info("Server stopped.")

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
    log.info("Starting server on http://%s:%s...", host, port)
    time.sleep(2)

    # Open browser
    url = f"http://{host}:{port}"
    log.info("Opening %s in browser...", url)
    webbrowser.open(url)

    # Keep main thread alive until Ctrl+C
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        log.info("Shutting down...")
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
        log.error(
            "No free port available starting from %s. "
            "Close other openVideoAI instances and retry.", args.port
        )
        sys.exit(1)

    log.info("=" * 50)
    log.info("openVideoAI - Local AI Video Enhancement")
    log.info("=" * 50)
    log.info("Mode: %s", args.mode)
    log.info("Host: %s", args.host)
    if effective_port != args.port:
        log.info("Port: %s (in use — using %s)", args.port, effective_port)
    else:
        log.info("Port: %s", effective_port)
    log.info("URL: http://%s:%s", args.host, effective_port)

    if args.mode == "window":
        run_desktop_window(args.host, effective_port, args.width, args.height)
    elif args.mode == "browser":
        run_browser_mode(args.host, effective_port)
    else:
        run_server(args.host, effective_port)

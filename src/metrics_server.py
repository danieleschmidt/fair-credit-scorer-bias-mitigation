"""Simple HTTP server for health checks and metrics endpoints."""

import json
import logging
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

from src.health_check import get_prometheus_metrics, health_checker


class MetricsHandler(BaseHTTPRequestHandler):
    """HTTP handler for metrics and health endpoints."""

    def log_message(self, format, *args):
        """Override to use application logger."""
        logging.getLogger(__name__).info(format % args)

    def do_GET(self):
        """Handle GET requests."""
        if self.path == "/health":
            self._handle_health()
        elif self.path == "/ready":
            self._handle_readiness()
        elif self.path == "/metrics":
            self._handle_metrics()
        elif self.path == "/metrics/prometheus":
            self._handle_prometheus_metrics()
        else:
            self._handle_not_found()

    def _handle_health(self):
        """Handle health check endpoint."""
        try:
            health_status = health_checker.get_health_status()
            self._send_json_response(200, health_status)
        except Exception as e:
            self._send_json_response(500, {"status": "error", "error": str(e)})

    def _handle_readiness(self):
        """Handle readiness check endpoint."""
        try:
            readiness_status = health_checker.get_readiness_status()
            status_code = 200 if readiness_status["status"] == "ready" else 503
            self._send_json_response(status_code, readiness_status)
        except Exception as e:
            self._send_json_response(503, {"status": "not_ready", "error": str(e)})

    def _handle_metrics(self):
        """Handle metrics endpoint."""
        try:
            metrics = health_checker.get_metrics()
            self._send_json_response(200, metrics)
        except Exception as e:
            self._send_json_response(500, {"error": str(e)})

    def _handle_prometheus_metrics(self):
        """Handle Prometheus metrics endpoint."""
        try:
            metrics_text = get_prometheus_metrics()
            self._send_text_response(200, metrics_text, "text/plain")
        except Exception as e:
            self._send_json_response(500, {"error": str(e)})

    def _handle_not_found(self):
        """Handle 404 not found."""
        self._send_json_response(404, {"error": "Not found"})

    def _send_json_response(self, status_code: int, data: dict):
        """Send JSON response."""
        self.send_response(status_code)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        response = json.dumps(data, indent=2)
        self.wfile.write(response.encode())

    def _send_text_response(self, status_code: int, data: str, content_type: str = "text/plain"):
        """Send text response."""
        self.send_response(status_code)
        self.send_header('Content-type', content_type)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(data.encode())


class MetricsServer:
    """HTTP server for metrics and health endpoints."""

    def __init__(self, host: str = "0.0.0.0", port: int = 8080):
        self.host = host
        self.port = port
        self.server = None
        self.thread = None
        self.logger = logging.getLogger(__name__)

    def start(self):
        """Start the metrics server in a background thread."""
        if self.server is not None:
            self.logger.warning("Metrics server already running")
            return

        try:
            self.server = HTTPServer((self.host, self.port), MetricsHandler)
            self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
            self.thread.start()
            self.logger.info(f"Metrics server started on {self.host}:{self.port}")
        except Exception as e:
            self.logger.error(f"Failed to start metrics server: {e}")
            raise

    def stop(self):
        """Stop the metrics server."""
        if self.server is None:
            return

        try:
            self.server.shutdown()
            self.server.server_close()
            if self.thread and self.thread.is_alive():
                self.thread.join(timeout=5)
            self.server = None
            self.thread = None
            self.logger.info("Metrics server stopped")
        except Exception as e:
            self.logger.error(f"Error stopping metrics server: {e}")

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()


# Default metrics server instance
default_metrics_server = MetricsServer()


def start_metrics_server(host: str = "0.0.0.0", port: int = 8080):
    """Start the default metrics server."""
    global default_metrics_server
    default_metrics_server = MetricsServer(host, port)
    default_metrics_server.start()
    return default_metrics_server


def stop_metrics_server():
    """Stop the default metrics server."""
    if default_metrics_server:
        default_metrics_server.stop()

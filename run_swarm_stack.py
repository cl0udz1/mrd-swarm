# -*- coding: utf-8 -*-
"""
run_swarm_stack.py — Unified Master Launcher for Decoupled Drone Swarm Stack

Orchestrates:
1. Sub-Agent 1: Headless MuJoCo Data-Oriented ECS Physics Simulation (100 Hz)
2. Sub-Agent 2: Real-Time WebSocket Telemetry Server (ws://127.0.0.1:8765 @ 60 Hz)
3. Sub-Agent 3: HTTP Static Server for Cinematic 3D WebGL Visualizer (http://127.0.0.1:8080)
"""

from __future__ import annotations
import argparse
import asyncio
import http.server
import os
import socketserver
import sys
import threading
import time
import webbrowser
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR))

from src.server import SwarmTelemetryServer, start_http_server, WS_HOST, WS_PORT, HTTP_PORT


def main():
    parser = argparse.ArgumentParser(description="MRD-SWARM Decoupled Dual-Stack Simulation Runner")
    parser.add_argument("--no-browser", action="store_true", help="Do not automatically launch the web browser")
    parser.add_argument("--ws-port", type=int, default=WS_PORT, help=f"WebSocket streaming port (default: {WS_PORT})")
    parser.add_argument("--http-port", type=int, default=HTTP_PORT, help=f"HTTP visualizer port (default: {HTTP_PORT})")
    args = parser.parse_args()

    print("=" * 95)
    print("  MRD-SWARM: DECOUPLED ECS PHYSICS + REAL-TIME 60Hz WEBGL VISUALIZER STACK")
    print("=" * 95)

    visualizer_dir = str(PROJECT_DIR / "visualizer")
    if not os.path.exists(visualizer_dir):
        print(f"Error: Visualizer directory not found at {visualizer_dir}")
        sys.exit(1)

    # 1. Start HTTP Server for 3D Visualizer Frontend
    http_thread = threading.Thread(
        target=start_http_server,
        args=(visualizer_dir, args.http_port),
        daemon=True,
    )
    http_thread.start()

    # 2. Auto-open Web Browser
    if not args.no_browser:
        url = f"http://127.0.0.1:{args.http_port}"
        print(f"[LAUNCHER] Opening 3D Visualizer in browser: {url}")
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()

    # 3. Start Telemetry Server & ECS Physics Step Loop
    server = SwarmTelemetryServer(host=WS_HOST, port=args.ws_port)
    print(f"[LAUNCHER] Physics Engine running in headless ECS mode (100 Hz)")
    print(f"[LAUNCHER] Telemetry Bridge streaming at 60 Hz on ws://{WS_HOST}:{args.ws_port}")
    print(f"[LAUNCHER] Press Ctrl+C to terminate the stack.\n")

    try:
        asyncio.run(server.run())
    except KeyboardInterrupt:
        print("\n[LAUNCHER] Shutting down simulation stack gracefully.")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
server.py — Real-Time Telemetry WebSocket Bridge & HTTP Visualizer Server

Provides:
- 60 Hz Async WebSocket Telemetry Streaming (ws://127.0.0.1:8765)
- Multi-client Broadcast (SE(3) poses, target telemetry, RF mesh links, uncertainty)
- Built-in HTTP Static File Server for 3D Visualizer (http://127.0.0.1:8080)
"""

from __future__ import annotations
import asyncio
import http.server
import json
import os
import socketserver
import sys
import threading
import time
from pathlib import Path
from typing import Set, Dict, Any

import websockets

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from src.ecs.world import ECSWorld

WS_HOST = "127.0.0.1"
WS_PORT = 8765
HTTP_PORT = 8080

OBSTACLES = [
    {"name": "Skyscraper Alpha", "pos": [0.0, 0.0, 7.0], "size": [4.0, 4.0, 7.0], "height": 14.0, "color": "#0ea5e9"},
    {"name": "Complex Bravo", "pos": [-14.0, 12.0, 3.0], "size": [6.0, 4.0, 3.0], "height": 6.0, "color": "#38bdf8"},
    {"name": "Silo Charlie", "pos": [-16.0, -14.0, 4.0], "size": [2.5, 2.5, 4.0], "height": 8.0, "color": "#64748b"},
    {"name": "Depot Delta", "pos": [15.0, -15.0, 2.5], "size": [7.0, 5.0, 2.5], "height": 5.0, "color": "#475569"},
    {"name": "Substation Echo", "pos": [14.0, 14.0, 2.0], "size": [4.5, 4.5, 2.0], "height": 4.0, "color": "#f59e0b"},
    {"name": "Radar Pylon Foxtrot", "pos": [22.0, 0.0, 5.0], "size": [1.5, 1.5, 5.0], "height": 10.0, "color": "#a855f7"},
    {"name": "Security Tower Golf", "pos": [-22.0, 0.0, 6.0], "size": [1.5, 1.5, 6.0], "height": 12.0, "color": "#ef4444"},
    {"name": "Skybridge Hotel", "pos": [0.0, -18.0, 4.5], "size": [10.0, 2.0, 1.0], "height": 7.0, "color": "#10b981"},
]


class SwarmTelemetryServer:
    """
    Asynchronous 60 Hz Telemetry Streaming Bridge.
    """

    def __init__(self, host: str = WS_HOST, port: int = WS_PORT):
        self.host = host
        self.port = port
        self.clients: Set[Any] = set()
        self.world = ECSWorld(obstacles=OBSTACLES, seed=42)
        self.running = False

    async def register_client(self, websocket):
        self.clients.add(websocket)
        print(f"[WS SERVER] Client connected: {websocket.remote_address} (Total: {len(self.clients)})")

        # Send initial world environment metadata (8 buildings, bounds)
        init_payload = {
            "type": "WORLD_METADATA",
            "theater_bounds": [-25, 25, -25, 25],
            "buildings": OBSTACLES,
        }
        await websocket.send(json.dumps(init_payload))

        try:
            async for message in websocket:
                try:
                    cmd = json.loads(message)
                    action = cmd.get("action")
                    if action == "TRIGGER_JAMMING":
                        state = self.world.trigger_jamming()
                        print(f"[TACTICAL CMD] EW Jamming State: {state}")
                    elif action == "TRIGGER_SMOKE":
                        tid = cmd.get("target_id", 0)
                        self.world.trigger_smoke(tid)
                        print(f"[TACTICAL CMD] Smoke Countermeasure Deployed by Target {tid}")
                    elif action == "TRIGGER_PINCER":
                        self.world.trigger_pincer()
                        print("[TACTICAL CMD] Swarm Executing Coordinated Pincer Ambush")
                    elif action == "TRIGGER_RTB":
                        did = cmd.get("drone_id", 1)
                        self.world.trigger_rtb(did)
                        print(f"[TACTICAL CMD] Drone {did} Initiating Emergency Rooftop RTB")
                    elif action == "SET_DOCTRINE":
                        doctrine = cmd.get("doctrine", "DEEPSEEK_ADAPTIVE")
                        doc_name = self.world.set_tactical_doctrine(doctrine)
                        print(f"[TACTICAL CMD] Swarm Tactical Doctrine switched to: {doc_name}")
                    elif action == "OPERATOR_COMMAND":
                        text = cmd.get("command", "").strip()
                        if text:
                            self.world.ai_commander.submit_operator_command(text)
                            print(f"[OPERATOR UPLINK] Directive transmitted to DeepSeek Commander: '{text}'")
                    elif action == "TRIGGER_VISION_SCAN":
                        did = cmd.get("drone_id", 1)
                        frame = self.world._generate_recon_camera_frame(drone_id=did)
                        self.world.vision_recon.request_vision_analysis(frame, drone_id=did, force=True)
                        print(f"[TACTICAL CMD] Priority DeepSeek Vision Recon Scan Triggered for D{did}")
                except Exception as e:
                    print("[WS ERROR] Command processing failed:", e)
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            self.clients.remove(websocket)
            print(f"[WS SERVER] Client disconnected (Remaining: {len(self.clients)})")

    async def broadcast_loop(self):
        """Runs the 100 Hz ECS simulation step and streams at 60 Hz."""
        sim_dt = 0.01  # 100 Hz simulation
        target_fps = 60.0
        frame_interval = 1.0 / target_fps
        last_frame_time = time.perf_counter()

        print(f"[WS SERVER] Starting 100 Hz ECS Simulation & 60 Hz Telemetry Stream...")

        while self.running:
            start_t = time.perf_counter()

            # Step ECS Simulation
            telemetry = self.world.step()

            # Broadcast to connected clients at 60 Hz
            now = time.perf_counter()
            if now - last_frame_time >= frame_interval:
                if self.clients:
                    payload = json.dumps(telemetry)
                    # Broadcast to all connected clients
                    await asyncio.gather(
                        *[client.send(payload) for client in self.clients],
                        return_exceptions=True,
                    )
                last_frame_time = now

            # Maintain simulation clock timing
            elapsed = time.perf_counter() - start_t
            sleep_time = max(0.001, sim_dt - elapsed)
            await asyncio.sleep(sleep_time)

    async def run(self):
        self.running = True
        print(f"[WS SERVER] Initializing WebSocket Server on ws://{self.host}:{self.port} ...")
        async with websockets.serve(self.register_client, self.host, self.port):
            await self.broadcast_loop()


def start_http_server(directory: str, port: int = HTTP_PORT):
    """Starts a background HTTP server for the 3D visualizer web assets."""
    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=directory, **kwargs)
        def log_message(self, format, *args):
            pass  # Suppress access logs

    with socketserver.TCPServer(("", port), Handler) as httpd:
        print(f"[HTTP SERVER] 3D Visualizer Frontend live at: http://127.0.0.1:{port}")
        httpd.serve_forever()


def main():
    visualizer_dir = str(PROJECT_DIR / "visualizer")
    os.makedirs(visualizer_dir, exist_ok=True)

    # Start HTTP server in a daemon thread
    http_thread = threading.Thread(target=start_http_server, args=(visualizer_dir, HTTP_PORT), daemon=True)
    http_thread.start()

    # Start Async WebSocket Server
    server = SwarmTelemetryServer(host=WS_HOST, port=WS_PORT)
    try:
        asyncio.run(server.run())
    except KeyboardInterrupt:
        print("\n[SERVER] Shutdown signal received. Exiting.")


if __name__ == "__main__":
    main()

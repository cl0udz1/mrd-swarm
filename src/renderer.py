"""
renderer.py — Headless Offscreen Rendering & MP4 Mission Video Report Synthesis

Supports:
    - CPU-only headless rendering via osmesa/egl
    - Multi-camera capture (overhead tactical + dual drone FPV streams)
    - Telemetry HUD overlay (flight paths, bounding boxes, swarm status, AI commands)
    - 3-Panel Split-Screen Video Compositing (Tactical Left + Drone 1 FPV + Drone 2 Flanker FPV)
"""

from __future__ import annotations

import os
import math
import numpy as np
from numpy.typing import NDArray
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
from PIL import Image, ImageDraw, ImageFont

import mujoco

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

try:
    import mediapy
    HAS_MEDIAPY = True
except ImportError:
    HAS_MEDIAPY = False


@dataclass
class CameraConfig:
    """Configuration for an offscreen camera."""
    name: str
    width: int = 640
    height: int = 480
    lookat: NDArray[np.float64] = field(default_factory=lambda: np.array([0.0, 0.0, 1.0]))
    distance: float = 15.0
    azimuth: float = 90.0
    elevation: float = -60.0
    fovy: float = 60.0


@dataclass
class HUDOverlay:
    """Data for rendering a heads-up display overlay."""
    mission_time: float = 0.0
    drone_positions: Dict[int, NDArray[np.float64]] = field(default_factory=dict)
    drone_velocities: Dict[int, NDArray[np.float64]] = field(default_factory=dict)
    drone_battery_pct: Dict[int, float] = field(default_factory=dict)
    drone_headings: Dict[int, float] = field(default_factory=dict)
    drone_roles: Dict[int, str] = field(default_factory=dict)
    target_positions: Dict[int, NDArray[np.float64]] = field(default_factory=dict)
    target_detected: Dict[int, bool] = field(default_factory=dict)
    detections: List[Dict[str, Any]] = field(default_factory=list)
    flight_trails: Dict[int, List[NDArray[np.float64]]] = field(default_factory=dict)
    active_mesh_links: List[Tuple[int, int]] = field(default_factory=list)
    uncertainty_pct: float = 100.0
    active_drones: int = 4
    total_drones: int = 4


class HeadlessRenderer:
    """
    Headless offscreen renderer for MuJoCo scenes.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        tactical_cam: CameraConfig | None = None,
        fpv_width: int = 640,
        fpv_height: int = 360,
    ):
        self.model = model
        self.fpv_width = fpv_width
        self.fpv_height = fpv_height

        self.tactical_cfg = tactical_cam or CameraConfig(
            name="tactical",
            width=1280,
            height=720,
            lookat=np.array([0.0, 0.0, 2.0]),
            distance=50.0,
            azimuth=50.0,
            elevation=-55.0,
            fovy=60.0,
        )

        self.tactical_renderer = mujoco.Renderer(
            model,
            width=self.tactical_cfg.width,
            height=self.tactical_cfg.height,
        )
        self.fpv_renderer = mujoco.Renderer(
            model,
            width=self.fpv_width,
            height=self.fpv_height,
        )

    def render_tactical(self, data: mujoco.MjData, azimuth_offset: float = 0.0) -> NDArray[np.uint8]:
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = self.tactical_cfg.lookat
        cam.distance = self.tactical_cfg.distance
        cam.azimuth = self.tactical_cfg.azimuth + azimuth_offset
        cam.elevation = self.tactical_cfg.elevation

        self.tactical_renderer.update_scene(data, cam)
        return self.tactical_renderer.render()

    def render_drone_fpv(
        self,
        data: mujoco.MjData,
        drone_pos: np.ndarray,
        drone_quat: np.ndarray,
        look_distance: float = 6.0,
    ) -> NDArray[np.uint8]:
        from .physics import quat_to_rotation_matrix
        R = quat_to_rotation_matrix(drone_quat)
        fwd = R[:, 0]

        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = drone_pos + fwd * look_distance
        cam.distance = 0.8
        cam.elevation = -10.0

        self.fpv_renderer.update_scene(data, cam)
        return self.fpv_renderer.render()

    def close(self) -> None:
        self.tactical_renderer.close()
        self.fpv_renderer.close()


class VideoReportGenerator:
    """
    Generates high-definition MP4 mission video reports.
    Supports 3-Panel Split-Screen: Tactical (Left, 720p) + Drone 1 FPV (Top-Right) + Drone 2 FPV (Bottom-Right).
    """

    def __init__(
        self,
        output_path: str = "output/dynamic_swarm_mission.mp4",
        fps: int = 50,
    ):
        self.output_path = output_path
        self.fps = fps
        self.frames: List[NDArray[np.uint8]] = []

    def compose_tri_panel_frame(
        self,
        tactical_rgb: NDArray[np.uint8],
        fpv1_rgb: NDArray[np.uint8],
        fpv2_rgb: NDArray[np.uint8],
        hud: HUDOverlay,
        active_directive_d1: str = "",
        active_directive_d2: str = "",
    ) -> NDArray[np.uint8]:
        """
        Composes a 1920x720 3-panel split-screen video frame.
        """
        # Canvas: 1920 x 720
        canvas = np.zeros((720, 1920, 3), dtype=np.uint8)

        # ── 1. Draw Left Tactical Frame (1280 x 720) ───────────────────────────
        img_tac = Image.fromarray(tactical_rgb).convert("RGBA")
        overlay_tac = Image.new("RGBA", img_tac.size, (0, 0, 0, 0))
        draw_tac = ImageDraw.Draw(overlay_tac)
        wt, ht = img_tac.size

        # Draw Glowing Cyan Mesh Links
        for id_a, id_b in hud.active_mesh_links:
            if id_a < id_b and id_a in hud.drone_positions and id_b in hud.drone_positions:
                pa = hud.drone_positions[id_a]
                pb = hud.drone_positions[id_b]
                px_a = int(np.clip(wt/2 + (pa[0] * 0.7 - pa[1] * 0.7) * (wt/55.0), 0, wt-1))
                py_a = int(np.clip(ht/2 + (pa[0] * 0.4 + pa[1] * 0.4 - pa[2] * 0.8) * (ht/55.0), 0, ht-1))
                px_b = int(np.clip(wt/2 + (pb[0] * 0.7 - pb[1] * 0.7) * (wt/55.0), 0, wt-1))
                py_b = int(np.clip(ht/2 + (pb[0] * 0.4 + pb[1] * 0.4 - pb[2] * 0.8) * (ht/55.0), 0, ht-1))
                draw_tac.line([(px_a, py_a), (px_b, py_b)], fill=(0, 240, 255, 220), width=2)

        # Draw Drones
        drone_palette = [(50, 160, 255), (255, 80, 50), (40, 220, 120), (200, 80, 255)]
        for d_id, dp in hud.drone_positions.items():
            px = int(np.clip(wt/2 + (dp[0] * 0.7 - dp[1] * 0.7) * (wt/55.0), 0, wt-1))
            py = int(np.clip(ht/2 + (dp[0] * 0.4 + dp[1] * 0.4 - dp[2] * 0.8) * (ht/55.0), 0, ht-1))
            role_txt = hud.drone_roles.get(d_id, "EXPLORER")
            draw_tac.ellipse([px-11, py-11, px+11, py+11], fill=drone_palette[d_id] + (230,), outline=(255, 255, 255, 255), width=2)
            draw_tac.text((px + 14, py - 8), f"D{d_id} [{role_txt[:8]}]", fill=(240, 245, 255, 255))

        # Draw Dynamic Targets (Red Diamonds)
        for t_id, tp in hud.target_positions.items():
            px = int(np.clip(wt/2 + (tp[0] * 0.7 - tp[1] * 0.7) * (wt/55.0), 0, wt-1))
            py = int(np.clip(ht/2 + (tp[0] * 0.4 + tp[1] * 0.4 - tp[2] * 0.8) * (ht/55.0), 0, ht-1))
            is_seen = hud.target_detected.get(t_id, False)
            color = (255, 50, 50, 240) if is_seen else (180, 180, 180, 200)
            draw_tac.polygon([(px, py-13), (px+13, py), (px, py+13), (px-13, py)], fill=color, outline=(255, 255, 255, 255))
            draw_tac.text((px + 15, py - 7), f"HVT-{t_id}", fill=(255, 220, 80, 255))

        # Tactical HUD Panel
        draw_tac.rectangle([15, 15, 520, 150], fill=(12, 18, 28, 235), outline=(0, 220, 255, 255))
        draw_tac.text((25, 22), "MRD-SWARM: CLOSED-LOOP 3D REACTIVE SWARM", fill=(255, 255, 255, 255))
        draw_tac.text((25, 44), f"Mission Time: {hud.mission_time:5.2f} s | Active Drones: {hud.active_drones}/{hud.total_drones}", fill=(200, 220, 245, 255))
        draw_tac.text((25, 64), f"3D Voxel Uncertainty: {hud.uncertainty_pct:5.1f}% | RF Mesh Links: {len(hud.active_mesh_links)}", fill=(0, 240, 255, 255))
        draw_tac.text((25, 84), f"HVT Sightings & Detections: {len(hud.detections)}", fill=(100, 255, 120, 255))
        draw_tac.text((25, 104), f"D1 Role: {hud.drone_roles.get(1, 'EXPLORER')} | D2 Role: {hud.drone_roles.get(2, 'FLANKER')}", fill=(255, 220, 100, 255))
        draw_tac.text((25, 124), f"D3 Comms Relay: Z={hud.drone_positions.get(3, np.zeros(3))[2]:.1f}m (High-Altitude Anchor)", fill=(200, 150, 255, 255))

        composed_tac = Image.alpha_composite(img_tac, overlay_tac).convert("RGB")
        canvas[:, :1280] = np.array(composed_tac)

        # ── 2. Draw Top-Right Panel: Drone 1 FPV (640 x 360) ───────────────────
        img_fpv1 = Image.fromarray(fpv1_rgb).resize((640, 360)).convert("RGBA")
        overlay_fpv1 = Image.new("RGBA", img_fpv1.size, (0, 0, 0, 0))
        draw_fpv1 = ImageDraw.Draw(overlay_fpv1)
        w1, h1 = img_fpv1.size
        cx1, cy1 = w1 // 2, h1 // 2

        # Optical Crosshairs
        draw_fpv1.line([(cx1 - 20, cy1), (cx1 - 5, cy1)], fill=(0, 255, 120, 240), width=2)
        draw_fpv1.line([(cx1 + 5, cy1), (cx1 + 20, cy1)], fill=(0, 255, 120, 240), width=2)
        draw_fpv1.line([(cx1, cy1 - 20), (cx1, cy1 - 5)], fill=(0, 255, 120, 240), width=2)
        draw_fpv1.line([(cx1, cy1 + 5), (cx1, cy1 + 20)], fill=(0, 255, 120, 240), width=2)

        # Status Bar
        draw_fpv1.rectangle([0, h1 - 30, w1, h1], fill=(10, 15, 25, 230))
        draw_fpv1.text((10, h1 - 22), f"DRONE 1 [FAST_INTERCEPTOR] | {hud.drone_roles.get(1, 'TRACKER')} | {active_directive_d1[:32]}", fill=(220, 235, 255, 255))

        composed_fpv1 = Image.alpha_composite(img_fpv1, overlay_fpv1).convert("RGB")
        canvas[:360, 1280:] = np.array(composed_fpv1)

        # ── 3. Draw Bottom-Right Panel: Drone 2 Flanker FPV (640 x 360) ────────
        img_fpv2 = Image.fromarray(fpv2_rgb).resize((640, 360)).convert("RGBA")
        overlay_fpv2 = Image.new("RGBA", img_fpv2.size, (0, 0, 0, 0))
        draw_fpv2 = ImageDraw.Draw(overlay_fpv2)
        w2, h2 = img_fpv2.size
        cx2, cy2 = w2 // 2, h2 // 2

        draw_fpv2.line([(cx2 - 20, cy2), (cx2 + 20, cy2)], fill=(255, 180, 40, 240), width=1)
        draw_fpv2.line([(cx2, cy2 - 20), (cx2, cy2 + 20)], fill=(255, 180, 40, 240), width=1)

        draw_fpv2.rectangle([0, h2 - 30, w2, h2], fill=(10, 15, 25, 230))
        draw_fpv2.text((10, h2 - 22), f"DRONE 2 [THERMAL_FLANKER] | {hud.drone_roles.get(2, 'FLANKER')} | {active_directive_d2[:32]}", fill=(220, 235, 255, 255))

        composed_fpv2 = Image.alpha_composite(img_fpv2, overlay_fpv2).convert("RGB")
        canvas[360:, 1280:] = np.array(composed_fpv2)

        # ── 4. Dividers & Labels ───────────────────────────────────────────────
        img_final = Image.fromarray(canvas)
        draw_final = ImageDraw.Draw(img_final)
        draw_final.line([(1280, 0), (1280, 720)], fill=(80, 100, 130), width=3)
        draw_final.line([(1280, 360), (1920, 360)], fill=(80, 100, 130), width=2)
        draw_final.text((1290, 10), "DRONE 1 PRIMARY RECON FPV", fill=(200, 220, 255))
        draw_final.text((1290, 370), "DRONE 2 TACTICAL FLANKER FPV", fill=(200, 220, 255))

        return np.array(img_final)

    def add_frame(self, frame: NDArray[np.uint8]) -> None:
        self.frames.append(frame)

    def save_video(self) -> str:
        import imageio
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        imageio.mimsave(self.output_path, self.frames, fps=self.fps, quality=9)
        return self.output_path

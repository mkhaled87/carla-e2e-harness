"""Interactive Pygame HUD & Multi-Camera Dashboard for E2E Driving Models in CARLA.

Displays:
1. Multi-Camera Sensor Wall (nuScenes surround 6-cam / QwenDrive & Alpamayo 7-cam).
2. Model Inputs: Navigation Intent, Speed, Kinematics History, Task Prompt.
3. Model Outputs: Predicted 2D Trajectory Corridor, Actuation Bars (Throttle, Brake, Steer).
4. Explainable Driving: Chain-of-Thought (CoT) / Chain-of-Causation (CoC) natural language reasoning.
"""

from __future__ import annotations

import math
import os
import sys
from typing import Any
import numpy as np

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "hide")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")


class NullDisplay:
    """Headless dummy display when running in non-GUI or CI environments."""
    def draw(self, **_: Any) -> bool:
        return True
    def close(self) -> None:
        pass


class E2EDashboardDisplay:
    """Rich interactive Pygame dashboard for CARLA E2E models."""

    WIDTH = 1600
    HEIGHT = 900
    CAM_HEIGHT = 480
    BOTTOM_HEIGHT = HEIGHT - CAM_HEIGHT

    # Colors (Dark Theme)
    BG_COLOR = (12, 15, 20)
    PANEL_BG = (20, 24, 32)
    BORDER_COLOR = (36, 44, 58)
    TEXT_MAIN = (240, 244, 250)
    TEXT_MUTED = (140, 150, 170)
    ACCENT_CYAN = (0, 210, 255)
    ACCENT_GREEN = (46, 213, 115)
    ACCENT_RED = (255, 71, 87)
    ACCENT_YELLOW = (255, 186, 0)
    ACCENT_PURPLE = (165, 94, 234)

    def __init__(self, title: str = "CARLA E2E AI Driving Cockpit") -> None:
        import pygame
        self.pygame = pygame
        pygame.init()
        pygame.font.init()

        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        pygame.display.set_caption(title)

        # Fonts
        self.font_title = pygame.font.SysFont("Helvetica,Arial", 20, bold=True)
        self.font_bold = pygame.font.SysFont("Helvetica,Arial", 16, bold=True)
        self.font_regular = pygame.font.SysFont("Helvetica,Arial", 14)
        self.font_small = pygame.font.SysFont("Helvetica,Arial", 12)
        self.font_mono = pygame.font.SysFont("Courier,Consolas,Monospace", 13)

        self._surface_cache: dict[str, Any] = {}

    def _handle_events(self) -> bool:
        for event in self.pygame.event.get():
            if event.type == self.pygame.QUIT:
                return False
            if event.type == self.pygame.KEYDOWN and event.key in (self.pygame.K_ESCAPE, self.pygame.K_q):
                return False
        return True

    def _convert_camera_image(self, img_rgb: np.ndarray, target_w: int, target_h: int):
        """Convert numpy RGB array (H, W, 3) into a scaled Pygame surface."""
        # pygame.surfarray requires transpose (W, H, 3)
        transposed = np.transpose(img_rgb, (1, 0, 2))
        surf = self.pygame.surfarray.make_surface(transposed)
        if surf.get_width() != target_w or surf.get_height() != target_h:
            surf = self.pygame.transform.smoothscale(surf, (target_w, target_h))
        return surf

    def draw(
        self,
        camera_images: dict[str, np.ndarray],
        model_name: str,
        high_level_command: str,
        text_prompt: str,
        current_speed_mps: float,
        target_speed_mps: float,
        throttle: float,
        brake: float,
        steer: float,
        cross_track_error: float,
        heading_error_rad: float,
        reasoning: str,
        waypoints: np.ndarray | None,
        step: int,
        sim_time_s: float | None = None,
        wall_time_s: float | None = None,
    ) -> bool:
        """Render complete frame."""
        if not self._handle_events():
            return False

        self.screen.fill(self.BG_COLOR)

        # 1. Render Multi-Camera Wall (Top 480 px)
        self._draw_camera_wall(camera_images)

        # 2. Render Left Panel: Model Input & Telemetry (0 to 450 px)
        self._draw_input_panel(
            model_name=model_name,
            cmd=high_level_command,
            prompt=text_prompt,
            speed=current_speed_mps,
            target_speed=target_speed_mps,
            step=step,
            sim_time_s=sim_time_s,
            wall_time_s=wall_time_s,
        )

        # 3. Render Center Panel: 2D Trajectory Corridor (450 to 950 px)
        self._draw_trajectory_panel(
            waypoints=waypoints,
            current_speed=current_speed_mps,
            cte=cross_track_error,
            he=heading_error_rad,
        )

        # 4. Render Right Panel: Actuation & Reasoning (950 to 1600 px)
        self._draw_output_panel(
            throttle=throttle,
            brake=brake,
            steer=steer,
            reasoning=reasoning,
        )

        self.pygame.display.flip()
        return True

    def _draw_camera_wall(self, camera_images: dict[str, np.ndarray]) -> None:
        """Render multi-camera grid layout."""
        if not camera_images:
            return

        cam_names = list(camera_images.keys())
        n_cams = len(cam_names)

        if n_cams <= 6:
            # 2 rows x 3 columns
            cols, rows = 3, 2
        else:
            # 2 rows x 4 columns
            cols, rows = 4, 2

        cell_w = self.WIDTH // cols
        cell_h = self.CAM_HEIGHT // rows

        # Preferred canonical ordering if present
        order_pref = [
            "CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_FRONT_WIDE", "CAM_FRONT_TELE",
            "front_cross_left", "front_wide", "front_cross_right", "front_tele",
            "CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT",
            "rear_left", "rear_tele", "rear_right",
        ]
        sorted_cams = sorted(cam_names, key=lambda c: order_pref.index(c) if c in order_pref else 99)

        for idx, name in enumerate(sorted_cams[:cols * rows]):
            c = idx % cols
            r = idx // cols
            x = c * cell_w
            y = r * cell_h

            img = camera_images.get(name)
            if img is not None and img.size > 0:
                surf = self._convert_camera_image(img, cell_w, cell_h)
                self.screen.blit(surf, (x, y))
            else:
                self.pygame.draw.rect(self.screen, (24, 28, 36), (x, y, cell_w, cell_h))

            # Overlay border & camera tag banner
            self.pygame.draw.rect(self.screen, self.BORDER_COLOR, (x, y, cell_w, cell_h), 1)

            banner = self.pygame.Surface((cell_w, 24), self.pygame.SRCALPHA)
            banner.fill((10, 14, 20, 200))
            self.screen.blit(banner, (x, y))

            # Camera name badge
            tag_color = self.ACCENT_CYAN if "front" in name.lower() else self.ACCENT_PURPLE
            self.pygame.draw.rect(self.screen, tag_color, (x, y, 4, 24))
            lbl = self.font_small.render(name.upper(), True, self.TEXT_MAIN)
            self.screen.blit(lbl, (x + 8, y + 4))

    def _draw_input_panel(
        self,
        model_name: str,
        cmd: str,
        prompt: str,
        speed: float,
        target_speed: float,
        step: int,
        sim_time_s: float | None = None,
        wall_time_s: float | None = None,
    ) -> None:
        """Render Left Panel: Model Input Information."""
        rect = self.pygame.Rect(10, self.CAM_HEIGHT + 10, 430, self.BOTTOM_HEIGHT - 20)
        self.pygame.draw.rect(self.screen, self.PANEL_BG, rect, border_radius=8)
        self.pygame.draw.rect(self.screen, self.BORDER_COLOR, rect, 1, border_radius=8)

        # Header
        title = self.font_title.render("MODEL INPUT CONTEXT", True, self.ACCENT_CYAN)
        self.screen.blit(title, (rect.x + 16, rect.y + 14))

        y = rect.y + 48

        # Model Architecture Badge
        lbl = self.font_small.render("AI ARCHITECTURE", True, self.TEXT_MUTED)
        self.screen.blit(lbl, (rect.x + 16, y))
        val = self.font_bold.render(model_name.upper(), True, self.TEXT_MAIN)
        self.screen.blit(val, (rect.x + 16, y + 16))

        # High Level Command Badge
        cmd_x = rect.x + 240
        lbl_cmd = self.font_small.render("NAVIGATION INTENT", True, self.TEXT_MUTED)
        self.screen.blit(lbl_cmd, (cmd_x, y))

        cmd_color = self.ACCENT_GREEN if "follow" in cmd.lower() else self.ACCENT_YELLOW
        if "stop" in cmd.lower():
            cmd_color = self.ACCENT_RED
        val_cmd = self.font_bold.render(cmd.upper(), True, cmd_color)
        self.screen.blit(val_cmd, (cmd_x, y + 16))

        y += 54

        # Speedometer section
        self.pygame.draw.line(self.screen, self.BORDER_COLOR, (rect.x + 16, y), (rect.x + rect.width - 16, y))
        y += 12

        speed_kmh = speed * 3.6
        tgt_kmh = target_speed * 3.6

        # Speed cards
        spd_card = self.pygame.Rect(rect.x + 16, y, 190, 68)
        self.pygame.draw.rect(self.screen, (16, 20, 26), spd_card, border_radius=6)
        lbl_sp = self.font_small.render("CURRENT SPEED", True, self.TEXT_MUTED)
        spd_val = self.font_title.render(f"{speed_kmh:.1f} km/h", True, self.ACCENT_CYAN)
        spd_sub = self.font_small.render(f"{speed:.2f} m/s", True, self.TEXT_MUTED)
        self.screen.blit(lbl_sp, (spd_card.x + 12, spd_card.y + 8))
        self.screen.blit(spd_val, (spd_card.x + 12, spd_card.y + 24))
        self.screen.blit(spd_sub, (spd_card.x + 12, spd_card.y + 48))

        tgt_card = self.pygame.Rect(rect.x + 224, y, 190, 68)
        self.pygame.draw.rect(self.screen, (16, 20, 26), tgt_card, border_radius=6)
        lbl_tg = self.font_small.render("TARGET SPEED", True, self.TEXT_MUTED)
        tgt_val = self.font_title.render(f"{tgt_kmh:.1f} km/h", True, self.ACCENT_GREEN)
        tgt_sub = self.font_small.render(f"{target_speed:.2f} m/s", True, self.TEXT_MUTED)
        self.screen.blit(lbl_tg, (tgt_card.x + 12, tgt_card.y + 8))
        self.screen.blit(tgt_val, (tgt_card.x + 12, tgt_card.y + 24))
        self.screen.blit(tgt_sub, (tgt_card.x + 12, tgt_card.y + 48))

        y += 82

        # Language Prompt box
        lbl_pr = self.font_small.render("TASK / SYSTEM PROMPT", True, self.TEXT_MUTED)
        self.screen.blit(lbl_pr, (rect.x + 16, y))
        y += 18

        prompt_box = self.pygame.Rect(rect.x + 16, y, rect.width - 32, 90)
        self.pygame.draw.rect(self.screen, (16, 20, 26), prompt_box, border_radius=6)
        self.pygame.draw.rect(self.screen, self.BORDER_COLOR, prompt_box, 1, border_radius=6)

        # Word wrap prompt text
        self._render_wrapped_text(
            surface=self.screen,
            text=prompt or "No active language prompt.",
            rect=self.pygame.Rect(prompt_box.x + 10, prompt_box.y + 8, prompt_box.width - 20, prompt_box.height - 16),
            font=self.font_regular,
            color=self.TEXT_MAIN,
        )

        # Step and actual time indicator at bottom-left of screen
        if sim_time_s is not None:
            time_text = f"Step: {step:03d}  |  Time: {sim_time_s:.2f}s"
            if wall_time_s is not None and wall_time_s > 0:
                time_text += f"  (Wall: {wall_time_s:.1f}s)"
        else:
            time_text = f"Simulation Step: {step}"
        step_lbl = self.font_small.render(time_text, True, self.TEXT_MUTED)
        self.screen.blit(step_lbl, (rect.x + 16, rect.bottom - 24))

    def _draw_trajectory_panel(
        self,
        waypoints: np.ndarray | None,
        current_speed: float,
        cte: float,
        he: float,
    ) -> None:
        """Render Center Panel: 2D Top-Down Trajectory Corridor."""
        rect = self.pygame.Rect(450, self.CAM_HEIGHT + 10, 480, self.BOTTOM_HEIGHT - 20)
        self.pygame.draw.rect(self.screen, self.PANEL_BG, rect, border_radius=8)
        self.pygame.draw.rect(self.screen, self.BORDER_COLOR, rect, 1, border_radius=8)

        # Header
        title = self.font_title.render("PREDICTED TRAJECTORY", True, self.ACCENT_CYAN)
        self.screen.blit(title, (rect.x + 16, rect.y + 14))

        # Coordinate canvas inside panel
        canvas = self.pygame.Rect(rect.x + 16, rect.y + 44, rect.width - 32, rect.height - 84)
        self.pygame.draw.rect(self.screen, (10, 13, 18), canvas, border_radius=6)
        self.pygame.draw.rect(self.screen, self.BORDER_COLOR, canvas, 1, border_radius=6)

        # Ego vehicle position on canvas (bottom-center facing UP)
        ego_cx = canvas.x + canvas.width // 2
        ego_cy = canvas.y + canvas.height - 35

        # Scale factor (pixels per meter): e.g. 50 meters range
        scale = 8.5  # 1 meter = 8.5 pixels

        # Draw grid reference lines
        for r_m in (10, 20, 30, 40):
            r_px = int(r_m * scale)
            if ego_cy - r_px > canvas.y:
                self.pygame.draw.line(self.screen, (22, 28, 38), (canvas.x, ego_cy - r_px), (canvas.right, ego_cy - r_px), 1)
                lbl_d = self.font_small.render(f"{r_m}m", True, (60, 75, 95))
                self.screen.blit(lbl_d, (canvas.x + 6, ego_cy - r_px - 14))

        # Draw vehicle icon (rectangle)
        veh_w, veh_l = int(2.0 * scale), int(4.8 * scale)
        veh_rect = self.pygame.Rect(ego_cx - veh_w // 2, ego_cy - veh_l // 2, veh_w, veh_l)
        self.pygame.draw.rect(self.screen, (50, 70, 95), veh_rect, border_radius=3)
        self.pygame.draw.rect(self.screen, self.ACCENT_CYAN, veh_rect, 1, border_radius=3)

        # Plot trajectory waypoints
        if waypoints is not None and len(waypoints) > 1:
            pts_px = []
            for pt in waypoints:
                # Local frame: +X is forward (UP on screen), +Y is right (RIGHT on screen)
                x_fwd, y_right = pt[0], pt[1]
                px = ego_cx + int(y_right * scale)
                py = ego_cy - int(x_fwd * scale)
                if canvas.collidepoint(px, py):
                    pts_px.append((px, py))

            if len(pts_px) > 1:
                self.pygame.draw.lines(self.screen, self.ACCENT_CYAN, False, pts_px, 3)
                for i, p in enumerate(pts_px):
                    dot_color = self.ACCENT_GREEN if i % 5 == 0 else self.ACCENT_CYAN
                    radius = 4 if i % 5 == 0 else 2
                    self.pygame.draw.circle(self.screen, dot_color, p, radius)

        # Metrics at bottom
        err_txt = f"Cross-Track Err: {cte:+.2f} m   Heading Err: {math.degrees(he):+.1f}°"
        lbl_err = self.font_small.render(err_txt, True, self.TEXT_MUTED)
        self.screen.blit(lbl_err, (rect.x + 16, rect.bottom - 26))

    def _draw_output_panel(
        self,
        throttle: float,
        brake: float,
        steer: float,
        reasoning: str,
    ) -> None:
        """Render Right Panel: Vehicle Actuation Bars & CoT Reasoning."""
        rect = self.pygame.Rect(940, self.CAM_HEIGHT + 10, 650, self.BOTTOM_HEIGHT - 20)
        self.pygame.draw.rect(self.screen, self.PANEL_BG, rect, border_radius=8)
        self.pygame.draw.rect(self.screen, self.BORDER_COLOR, rect, 1, border_radius=8)

        # Header
        title = self.font_title.render("OUTPUT ACTUATION & EXPLANATION", True, self.ACCENT_CYAN)
        self.screen.blit(title, (rect.x + 16, rect.y + 14))

        y = rect.y + 48

        # --- ACTUATION GAUGES ---
        bar_w = 175
        bar_h = 16

        # Throttle Bar
        self.screen.blit(self.font_small.render(f"THROTTLE: {throttle:.2f}", True, self.TEXT_MAIN), (rect.x + 16, y))
        th_bg = self.pygame.Rect(rect.x + 16, y + 18, bar_w, bar_h)
        self.pygame.draw.rect(self.screen, (26, 32, 42), th_bg, border_radius=4)
        th_fill = self.pygame.Rect(rect.x + 16, y + 18, int(bar_w * np.clip(throttle, 0.0, 1.0)), bar_h)
        self.pygame.draw.rect(self.screen, self.ACCENT_GREEN, th_fill, border_radius=4)

        # Brake Bar
        bx = rect.x + 215
        self.screen.blit(self.font_small.render(f"BRAKE: {brake:.2f}", True, self.TEXT_MAIN), (bx, y))
        br_bg = self.pygame.Rect(bx, y + 18, bar_w, bar_h)
        self.pygame.draw.rect(self.screen, (26, 32, 42), br_bg, border_radius=4)
        br_fill = self.pygame.Rect(bx, y + 18, int(bar_w * np.clip(brake, 0.0, 1.0)), bar_h)
        self.pygame.draw.rect(self.screen, self.ACCENT_RED, br_fill, border_radius=4)

        # Steer Bar (Bi-directional: center = 0)
        sx = rect.x + 415
        self.screen.blit(self.font_small.render(f"STEER: {steer:+.2f}", True, self.TEXT_MAIN), (sx, y))
        st_bg = self.pygame.Rect(sx, y + 18, bar_w, bar_h)
        self.pygame.draw.rect(self.screen, (26, 32, 42), st_bg, border_radius=4)
        center_x = sx + bar_w // 2
        self.pygame.draw.line(self.screen, self.TEXT_MUTED, (center_x, y + 16), (center_x, y + 18 + bar_h + 2), 2)
        steer_px = int((bar_w // 2) * np.clip(steer, -1.0, 1.0))
        if steer_px >= 0:
            st_fill = self.pygame.Rect(center_x, y + 18, steer_px, bar_h)
        else:
            st_fill = self.pygame.Rect(center_x + steer_px, y + 18, -steer_px, bar_h)
        self.pygame.draw.rect(self.screen, self.ACCENT_CYAN, st_fill, border_radius=4)

        y += 54
        self.pygame.draw.line(self.screen, self.BORDER_COLOR, (rect.x + 16, y), (rect.x + rect.width - 16, y))
        y += 12

        # --- REASONING / CHAIN-OF-THOUGHT BOX ---
        lbl_cot = self.font_bold.render("CHAIN-OF-THOUGHT / DRIVING REASONING", True, self.ACCENT_YELLOW)
        self.screen.blit(lbl_cot, (rect.x + 16, y))
        y += 22

        cot_box = self.pygame.Rect(rect.x + 16, y, rect.width - 32, rect.bottom - y - 16)
        self.pygame.draw.rect(self.screen, (10, 14, 20), cot_box, border_radius=6)
        self.pygame.draw.rect(self.screen, (40, 50, 68), cot_box, 1, border_radius=6)

        self._render_wrapped_text(
            surface=self.screen,
            text=reasoning or "Autonomous driving model computing trajectory...",
            rect=self.pygame.Rect(cot_box.x + 12, cot_box.y + 10, cot_box.width - 24, cot_box.height - 20),
            font=self.font_mono,
            color=(235, 240, 248),
            line_spacing=4,
        )

    def _render_wrapped_text(
        self,
        surface: Any,
        text: str,
        rect: Any,
        font: Any,
        color: tuple[int, int, int],
        line_spacing: int = 3,
    ) -> None:
        """Render multi-line text with word wrapping."""
        lines: list[str] = []
        for paragraph in text.splitlines():
            words = paragraph.split(" ")
            curr_line = ""
            for word in words:
                test_line = f"{curr_line} {word}".strip()
                if font.size(test_line)[0] < rect.width:
                    curr_line = test_line
                else:
                    if curr_line:
                        lines.append(curr_line)
                    curr_line = word
            if curr_line:
                lines.append(curr_line)

        y = rect.y
        for line in lines:
            if y + font.get_linesize() > rect.bottom:
                break
            rendered = font.render(line, True, color)
            surface.blit(rendered, (rect.x, y))
            y += font.get_linesize() + line_spacing

    def close(self) -> None:
        """Close Pygame window."""
        try:
            self.pygame.quit()
        except Exception:
            pass


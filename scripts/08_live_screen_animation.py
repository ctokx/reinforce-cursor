#!/usr/bin/env python3

import sys
import time
import argparse
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def lerp_color(c1, c2, t):
    t = max(0, min(1, t))
    return tuple(int(c1[i] + (c2[i] - c1[i]) * t) for i in range(3))


def speed_to_color(speed, max_speed):
    t = min(speed / max(max_speed, 1e-6), 1.0)
    if t < 0.5:
        return lerp_color((60, 120, 255), (60, 220, 100), t * 2)
    else:
        return lerp_color((60, 220, 100), (255, 80, 60), (t - 0.5) * 2)


def main():
    parser = argparse.ArgumentParser(
        description="Live 2D screen animation of BMDS trajectories"
    )
    parser.add_argument("--model", type=str, default=None,
                        help="Path to .d3 model file")
    parser.add_argument("--n", type=int, default=8,
                        help="Number of movements (default: 8)")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="Playback speed (0.5=slow, 2.0=fast)")
    parser.add_argument("--resolution", type=int, nargs=2,
                        default=[1920, 1080],
                        help="Virtual screen resolution (default: 1920 1080)")
    parser.add_argument("--window-scale", type=float, default=0.6,
                        help="Window size as fraction of resolution (default: 0.6)")
    parser.add_argument("--untrained", action="store_true",
                        help="Use untrained (random) policy")
    parser.add_argument("--both", action="store_true",
                        help="Side-by-side: trained vs untrained")
    parser.add_argument("--trail-length", type=int, default=80,
                        help="Trail length in points (default: 80)")
    parser.add_argument("--pause", type=float, default=5.0,
                        help="Pause between movements in seconds (default: 5.0)")
    parser.add_argument("--dark", action="store_true", default=True,
                        help="Dark background (default)")
    parser.add_argument("--save-gif", type=str, default=None,
                        help="Output GIF path (e.g. output/visualizations/live_animation.gif)")
    parser.add_argument("--gif-fps", type=int, default=20,
                        help="GIF frame rate (default: 20)")
    parser.add_argument("--gif-stride", type=int, default=2,
                        help="Capture every Nth frame for GIF (default: 2)")
    parser.add_argument("--auto-exit", action="store_true",
                        help="Exit automatically after rendering all movements")
    args = parser.parse_args()

    import pygame
    import d3rlpy

    from bmds.config import MODELS_DIR
    from bmds.synthesizer import BMDSSynthesizer

    print("=" * 60)
    print("BMDS — Live 2D Screen Animation")
    print("=" * 60)

    screen_res = tuple(args.resolution)
    win_w = int(screen_res[0] * args.window_scale)
    win_h = int(screen_res[1] * args.window_scale)
    scale_x = win_w / screen_res[0]
    scale_y = win_h / screen_res[1]


    if args.both:
        synth_trained = BMDSSynthesizer.load(
            model_path=args.model, screen_resolution=screen_res
        )
        synth_untrained = BMDSSynthesizer.load_untrained(
            screen_resolution=screen_res
        )
        total_width = win_w * 2 + 4
        print("Mode: SIDE-BY-SIDE (trained vs untrained)")
    else:
        if args.untrained:
            synth = BMDSSynthesizer.load_untrained(screen_resolution=screen_res)
            print("Mode: UNTRAINED (random)")
        else:
            synth = BMDSSynthesizer.load(
                model_path=args.model, screen_resolution=screen_res
            )
            print("Mode: TRAINED policy")
        total_width = win_w
        synth_trained = synth
        synth_untrained = None


    test_movements = [

        ((200, 200), (420, 310)),
        ((1600, 800), (1380, 650)),
        ((960, 540), (1100, 480)),

        ((100, 100), (800, 500)),
        ((1500, 800), (400, 200)),
        ((300, 800), (1200, 300)),

        ((100, 540), (1820, 540)),
        ((1820, 200), (100, 200)),

        ((400, 50), (400, 980)),
        ((1400, 980), (1400, 50)),

        ((80, 80), (1840, 1000)),
        ((1840, 80), (80, 1000)),

        ((700, 300), (1300, 750)),
        ((500, 700), (1400, 200)),
        ((960, 100), (200, 900)),
    ]
    if args.n > len(test_movements):
        rng = np.random.default_rng(42)
        while len(test_movements) < args.n:
            start = (int(rng.integers(50, 1870)), int(rng.integers(50, 1030)))
            end = (int(rng.integers(50, 1870)), int(rng.integers(50, 1030)))
            test_movements.append((start, end))
    test_movements = test_movements[:args.n]

    print(f"Virtual screen: {screen_res[0]}×{screen_res[1]}")
    print(f"Window: {win_w}×{win_h} ({args.window_scale:.0%} scale)")
    print(f"Movements: {args.n}")
    print(f"Speed: {args.speed}x\n")


    print("Pre-generating trajectories...")
    trajectories_main = []
    trajectories_alt = []

    for i, (start, end) in enumerate(test_movements):
        traj = synth_trained.generate(start=start, end=end, noise_seed=i)
        trajectories_main.append(np.array(traj))
        if args.both:
            traj_u = synth_untrained.generate(start=start, end=end, noise_seed=i)
            trajectories_alt.append(np.array(traj_u))
        print(f"  {i+1}/{len(test_movements)}: {start} -> {end} "
              f"({len(traj)} pts, {traj[-1][2]:.3f}s)")

    print(f"\nAll trajectories generated. Starting animation...")
    print("(Close the window or press ESC/Q to exit)\n")


    pygame.init()
    screen = pygame.display.set_mode((total_width, win_h))
    pygame.display.set_caption("BMDS — Synthetic Mouse Trajectory Animation")
    clock = pygame.time.Clock()
    capture_gif = args.save_gif is not None
    capture_stride = max(1, int(args.gif_stride))
    capture_index = 0
    gif_frames = []
    if capture_gif:
        args.auto_exit = True

    def maybe_capture_frame():
        nonlocal capture_index
        if not capture_gif:
            return
        if capture_index % capture_stride == 0:
            frame = pygame.surfarray.array3d(screen)
            frame = np.transpose(frame, (1, 0, 2))
            gif_frames.append(frame.copy())
        capture_index += 1


    BG = (18, 18, 24)
    GRID = (35, 35, 50)
    TEXT_COL = (200, 200, 220)
    START_COL = (80, 220, 120)
    END_COL = (255, 80, 80)
    CURSOR_COL = (255, 255, 255)
    DIVIDER = (60, 60, 80)
    LABEL_TRAINED = (100, 200, 255)
    LABEL_UNTRAINED = (255, 180, 80)

    try:
        font = pygame.font.SysFont("Consolas", 14)
        font_big = pygame.font.SysFont("Consolas", 18, bold=True)
        font_title = pygame.font.SysFont("Consolas", 22, bold=True)
    except Exception:
        font = pygame.font.Font(None, 16)
        font_big = pygame.font.Font(None, 20)
        font_title = pygame.font.Font(None, 24)

    def to_win(px, py, offset_x=0):
        return int(px * scale_x) + offset_x, int(py * scale_y)

    def draw_grid(surface, offset_x=0, w=win_w, h=win_h):
        for x in range(0, w, 50):
            pygame.draw.line(surface, GRID, (offset_x + x, 0),
                           (offset_x + x, h), 1)
        for y in range(0, h, 50):
            pygame.draw.line(surface, GRID, (offset_x, y),
                           (offset_x + w, y), 1)

    def draw_trajectory_frame(surface, traj_arr, frame_idx, start_px,
                              end_px, offset_x=0, label=None, label_col=None):

        sx, sy = to_win(*start_px, offset_x)
        pygame.draw.circle(surface, START_COL, (sx, sy), 8, 2)
        pygame.draw.circle(surface, START_COL, (sx, sy), 3)


        ex, ey = to_win(*end_px, offset_x)
        pygame.draw.circle(surface, END_COL, (ex, ey), 10, 2)
        pygame.draw.line(surface, END_COL, (ex - 6, ey), (ex + 6, ey), 2)
        pygame.draw.line(surface, END_COL, (ex, ey - 6), (ex, ey + 6), 2)


        if len(traj_arr) > 1:
            dx = np.diff(traj_arr[:, 0])
            dy = np.diff(traj_arr[:, 1])
            dt = np.diff(traj_arr[:, 2])
            dt = np.where(dt > 0, dt, 1e-6)
            speeds = np.sqrt(dx**2 + dy**2) / dt
            max_speed = max(np.max(speeds), 1)
        else:
            speeds = np.array([0])
            max_speed = 1


        trail_start = max(0, frame_idx - args.trail_length)
        for j in range(trail_start, min(frame_idx, len(traj_arr) - 1)):
            age = (frame_idx - j) / args.trail_length
            alpha = max(0, 1.0 - age)
            sp = speeds[j] if j < len(speeds) else 0
            base_col = speed_to_color(sp, max_speed)
            col = tuple(int(c * alpha) for c in base_col)
            p1 = to_win(traj_arr[j, 0], traj_arr[j, 1], offset_x)
            p2 = to_win(traj_arr[j + 1, 0], traj_arr[j + 1, 1], offset_x)
            width = max(1, int(3 * alpha))
            pygame.draw.line(surface, col, p1, p2, width)


        if frame_idx < len(traj_arr):
            cx, cy = to_win(traj_arr[frame_idx, 0],
                           traj_arr[frame_idx, 1], offset_x)
            pygame.draw.circle(surface, CURSOR_COL, (cx, cy), 6)
            pygame.draw.circle(surface, (0, 0, 0), (cx, cy), 3)


        if label:
            lbl = font_big.render(label, True, label_col or TEXT_COL)
            surface.blit(lbl, (offset_x + 10, 10))


    running = True
    mov_idx = 0

    while running and mov_idx < len(test_movements):
        start_px, end_px = test_movements[mov_idx]
        traj_main = trajectories_main[mov_idx]
        traj_alt = trajectories_alt[mov_idx] if args.both else None

        n_frames = len(traj_main)
        if args.both and traj_alt is not None:
            n_frames = max(n_frames, len(traj_alt))


        frame_idx = 0
        anim_start = time.time()

        while frame_idx < n_frames and running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_ESCAPE, pygame.K_q):
                        running = False
                    elif event.key == pygame.K_SPACE:

                        frame_idx = n_frames
                    elif event.key == pygame.K_r:

                        frame_idx = 0
                        anim_start = time.time()

            if not running:
                break


            screen.fill(BG)
            draw_grid(screen, 0, win_w, win_h)

            if args.both:
                draw_grid(screen, win_w + 4, win_w, win_h)

                pygame.draw.line(screen, DIVIDER, (win_w + 1, 0),
                               (win_w + 1, win_h), 3)


            fi_main = min(frame_idx, len(traj_main) - 1)
            main_label = "TRAINED" if not args.untrained else "UNTRAINED"
            main_col = LABEL_TRAINED if not args.untrained else LABEL_UNTRAINED
            draw_trajectory_frame(screen, traj_main, fi_main,
                                start_px, end_px, 0, main_label, main_col)


            if args.both and traj_alt is not None:
                fi_alt = min(frame_idx, len(traj_alt) - 1)
                draw_trajectory_frame(screen, traj_alt, fi_alt,
                                    start_px, end_px, win_w + 4,
                                    "UNTRAINED", LABEL_UNTRAINED)


            elapsed_sim = (traj_main[fi_main, 2]
                          if fi_main < len(traj_main) else traj_main[-1, 2])
            hud_lines = [
                f"Movement {mov_idx+1}/{len(test_movements)}",
                f"({start_px[0]},{start_px[1]}) -> ({end_px[0]},{end_px[1]})",
                f"Frame {fi_main+1}/{len(traj_main)} | t={elapsed_sim:.3f}s",
                f"[SPACE] skip  [R] replay  [Q] quit",
            ]
            y_off = win_h - len(hud_lines) * 20 - 10
            for line in hud_lines:
                txt = font.render(line, True, TEXT_COL)
                screen.blit(txt, (10, y_off))
                y_off += 20

            pygame.display.flip()
            maybe_capture_frame()


            frame_idx += 1
            sleep_time = (traj_main[min(fi_main + 1, len(traj_main) - 1), 2]
                         - traj_main[fi_main, 2]) / args.speed
            if sleep_time > 0:
                time.sleep(min(sleep_time, 0.1))
            clock.tick(120)


        if running:
            t_pause = time.time()
            while time.time() - t_pause < args.pause and running:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    elif event.type == pygame.KEYDOWN:
                        if event.key in (pygame.K_ESCAPE, pygame.K_q):
                            running = False
                        elif event.key == pygame.K_SPACE:
                            break
                clock.tick(30)

        mov_idx += 1


    if running:
        print("\nAll movements complete! Showing full trajectory overview...")
        screen.fill(BG)
        draw_grid(screen, 0, win_w, win_h)

        for traj_arr in trajectories_main:
            if len(traj_arr) > 1:
                dx = np.diff(traj_arr[:, 0])
                dy = np.diff(traj_arr[:, 1])
                dt = np.diff(traj_arr[:, 2])
                dt = np.where(dt > 0, dt, 1e-6)
                speeds = np.sqrt(dx**2 + dy**2) / dt
                max_speed = max(np.max(speeds), 1)
                for j in range(len(traj_arr) - 1):
                    col = speed_to_color(speeds[j], max_speed)
                    p1 = to_win(traj_arr[j, 0], traj_arr[j, 1])
                    p2 = to_win(traj_arr[j + 1, 0], traj_arr[j + 1, 1])
                    pygame.draw.line(screen, col, p1, p2, 2)


            s = to_win(traj_arr[0, 0], traj_arr[0, 1])
            e = to_win(traj_arr[-1, 0], traj_arr[-1, 1])
            pygame.draw.circle(screen, START_COL, s, 5, 2)
            pygame.draw.circle(screen, END_COL, e, 5, 2)

        title = font_title.render("BMDS — All Trajectories Overview", True, TEXT_COL)
        screen.blit(title, (10, 10))
        hint = font.render("Press any key or close window to exit", True, TEXT_COL)
        screen.blit(hint, (10, win_h - 25))
        pygame.display.flip()
        maybe_capture_frame()


        if not args.auto_exit:
            waiting = True
            while waiting:
                for event in pygame.event.get():
                    if event.type in (pygame.QUIT, pygame.KEYDOWN):
                        waiting = False
                clock.tick(15)

    if capture_gif and gif_frames:
        from PIL import Image

        output_path = Path(args.save_gif)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        images = [Image.fromarray(frame) for frame in gif_frames]
        frame_duration_ms = max(1, int(1000 / max(1, args.gif_fps)))
        images[0].save(
            output_path,
            save_all=True,
            append_images=images[1:],
            duration=frame_duration_ms,
            loop=0,
        )
        print(f"Saved GIF: {output_path} ({len(images)} frames @ {args.gif_fps} fps)")

    pygame.quit()
    print("Animation closed.")


if __name__ == "__main__":
    main()

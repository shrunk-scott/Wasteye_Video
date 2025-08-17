#!/usr/bin/env python
"""
YOLOEtest.py
- YOLOE + CLIP text prompts
- Unique counting with ByteTrack
- Two modes: Line-crossing OR Zone (polygon) entry
- On-screen editing:
    • Draw/Edit LINE: click two points (A then B)
    • Draw/Edit ZONE: left-click to add vertices; Right-click to close; 'z' undo last; 'x' clear
    • Prompts: press 'p' to enter edit mode, type comma-separated prompts on-screen, Enter to apply
- Save counts to spreadsheet: press 's' (saves .xlsx and .csv), also auto-saves on quit
- Overlay image: bottom-left with transparency via --overlay / --overlay_width / --overlay_margin
"""

import os
import sys
import time
import csv
import argparse
from collections import defaultdict
from datetime import datetime

import numpy as np
import cv2
import torch

# ------------------------- Args ----------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="YOLOE counter with on-screen editing + spreadsheets + overlay.")
    p.add_argument("--model", default="yoloe-11s-seg.pt", help="YOLOE prompted checkpoint")
    p.add_argument("--source", default="0", help="Webcam index (0/1/2) or path/URL")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--device", choices=["auto","mps","cpu"], default="auto")
    p.add_argument("--prompts", default= "person", help="Initial prompts (comma-separated)")
    p.add_argument("--save_prefix", default="counts_session", help="Base filename for spreadsheets")

    # --- Overlay options ---
    p.add_argument("--overlay", default="/Users/shrunk/Documents/YOLOE/outline.png",
                   help="Path to PNG/JPG to overlay at bottom-left (transparent PNG supported)")
    p.add_argument("--overlay_width", type=int, default=220,
                   help="Overlay target width in pixels (keeps aspect ratio; <=0 to keep original)")
    p.add_argument("--overlay_margin", type=int, default=12,
                   help="Margin from edges in pixels")
    return p.parse_args()


# ------------------------- Helpers -------------------------------------------
def pick_device(arg_device: str) -> str:
    if arg_device == "auto":
        return "mps" if torch.backends.mps.is_available() else "cpu"
    return arg_device

def center_of_box(xyxy):
    x1,y1,x2,y2 = xyxy
    return int((x1+x2)/2), int((y1+y2)/2)

def side_of_line(p, a, b):
    (x,y) = p; (x1,y1) = a; (x2,y2) = b
    return np.sign((x2-x1)*(y-y1) - (y2-y1)*(x-x1))

def poly_to_mask(poly_pts, w, h):
    mask = np.zeros((h, w), dtype=np.uint8)
    if poly_pts is not None and len(poly_pts) >= 3:
        cv2.fillPoly(mask, [np.array(poly_pts, dtype=np.int32)], 1)
    return mask

def timestamp_str():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

def safe_save_spreadsheets(save_prefix, totals_dict, mode, geometry_str, prompts_list, device, extra_notes=""):
    if not totals_dict:
        print("No counts to save yet; saving empty totals.")
    rows = []
    for cls_name, count in sorted(totals_dict.items()):
        rows.append({
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "class": cls_name,
            "count": int(count),
            "mode": mode,
            "geometry": geometry_str,
            "prompts": ", ".join(prompts_list),
            "device": device,
            "notes": extra_notes,
        })
    if not rows:
        rows = [{
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "class": "(none)",
            "count": 0,
            "mode": mode,
            "geometry": geometry_str,
            "prompts": ", ".join(prompts_list),
            "device": device,
            "notes": extra_notes,
        }]

    base = f"{save_prefix}_{timestamp_str()}"
    csv_path = f"{base}.csv"
    try:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"Saved CSV → {csv_path}")
    except Exception as e:
        print("Failed to save CSV:", e)

    try:
        import pandas as pd  # noqa
        df = pd.DataFrame(rows)
        xlsx_path = f"{base}.xlsx"
        df.to_excel(xlsx_path, index=False)
        print(f"Saved Excel → {xlsx_path}")
    except Exception as e:
        print("Pandas/openpyxl not available or failed to write .xlsx:", e)

def paste_overlay_rgba(dst_bgr, overlay_rgba, x, y):
    """
    Alpha-blend overlay_rgba (H,W,4) onto dst_bgr (H,W,3) with top-left at (x,y).
    Clips if it would go out of bounds.
    """
    H, W = dst_bgr.shape[:2]
    oh, ow = overlay_rgba.shape[:2]

    if x >= W or y >= H or x + ow <= 0 or y + oh <= 0:
        return

    x0 = max(x, 0); y0 = max(y, 0)
    x1 = min(x + ow, W); y1 = min(y + oh, H)

    ox0 = x0 - x; oy0 = y0 - y
    ox1 = ox0 + (x1 - x0); oy1 = oy0 + (y1 - y0)

    roi = dst_bgr[y0:y1, x0:x1]
    ov  = overlay_rgba[oy0:oy1, ox0:ox1]

    if ov.shape[2] == 4:
        overlay_bgr = ov[..., :3].astype(np.float32)
        alpha = (ov[..., 3:4].astype(np.float32)) / 255.0
    else:
        overlay_bgr = ov.astype(np.float32)
        alpha = np.ones_like(overlay_bgr[..., :1], dtype=np.float32)

    base = roi.astype(np.float32)
    blended = alpha * overlay_bgr + (1.0 - alpha) * base
    dst_bgr[y0:y1, x0:x1] = blended.astype(np.uint8)


# ------------------------- Main ----------------------------------------------
def main():
    args = parse_args()
    src = int(args.source) if args.source.isdigit() else args.source

    device = pick_device(args.device)
    print("="*70)
    print("Python:", sys.executable)
    print(f"Device: {device} (MPS available: {torch.backends.mps.is_available()})")
    print("Model :", args.model)

    try:
        from ultralytics import YOLOE
    except Exception as e:
        print("Failed to import ultralytics:", e)
        sys.exit(1)

    prompts = [s.strip() for s in args.prompts.split(",") if s.strip()]
    clip_ok = True
    if prompts:
        try:
            import clip  # noqa: F401
        except Exception:
            clip_ok = False
            print("WARNING: CLIP not importable. Prompts will NOT filter classes.")
            print("         pip install git+https://github.com/ultralytics/CLIP.git")

    model = YOLOE(args.model)
    if prompts and clip_ok:
        try:
            model.set_classes(prompts, model.get_text_pe(prompts))
            print("Text prompts set:", prompts)
        except Exception as e:
            print("WARNING: Failed to set prompts; running open-vocab. Err:", e)

    TRACKER_CFG = "bytetrack.yaml"
    totals = defaultdict(int)
    counted_tracks = set()
    last_side = {}
    in_zone = {}
    show_centers = False

    mode = "line"
    line_pts = []
    zone_pts = []
    drawing_zone = False

    prompt_edit = False
    prompt_buffer = ""

    window = "YOLOE Counter (GUI)"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    def on_mouse(event, x, y, flags, param):
        nonlocal line_pts, zone_pts, drawing_zone
        if mode == "line":
            if event == cv2.EVENT_LBUTTONDOWN:
                if len(line_pts) >= 2:
                    line_pts = []
                line_pts.append((x, y))
            elif event == cv2.EVENT_RBUTTONDOWN:
                line_pts = []
        else:
            if event == cv2.EVENT_LBUTTONDOWN:
                zone_pts.append((x, y))
                drawing_zone = True
            elif event == cv2.EVENT_RBUTTONDOWN:
                drawing_zone = False
            elif event == cv2.EVENT_MOUSEMOVE and drawing_zone:
                pass

    cv2.setMouseCallback(window, on_mouse)

    print("="*70)
    print("Hotkeys: q=quit | s=save | m=mode | r=reset | c=centers | p=prompt edit | z=undo (zone) | x=clear (zone)")
    print("Mouse (line):  left-click twice to set A->B, right-click to clear")
    print("Mouse (zone):  left-click to add vertex, right-click to close; 'z' undo, 'x' clear")
    print("="*70)

    t_last = time.time()

    def reset_counts():
        totals.clear()
        counted_tracks.clear()
        last_side.clear()
        in_zone.clear()
        print("Counts reset.")

    def apply_prompt_buffer():
        nonlocal prompts, prompt_buffer
        new_prompts = [s.strip() for s in prompt_buffer.split(",") if s.strip()]
        if len(new_prompts) == 0:
            print("Prompts cleared; running open-vocab.")
            prompts = []
            return
        if not clip_ok:
            print("CLIP not available; cannot apply text prompts.")
            return
        try:
            model.set_classes(new_prompts, model.get_text_pe(new_prompts))
            prompts = new_prompts
            print("Applied prompts:", prompts)
        except Exception as e:
            print("Failed to apply prompts:", e)

    # ----- Overlay image (optional) -----
    overlay_rgba = None
    if args.overlay and os.path.exists(args.overlay):
        ov = cv2.imread(args.overlay, cv2.IMREAD_UNCHANGED)
        if ov is None:
            print(f"WARNING: Could not read overlay image: {args.overlay}")
        else:
            if ov.ndim == 3 and ov.shape[2] == 3:
                alpha = 255 * np.ones((ov.shape[0], ov.shape[1], 1), dtype=ov.dtype)
                ov = np.concatenate([ov, alpha], axis=2)
            if args.overlay_width > 0 and ov.shape[1] != args.overlay_width:
                scale = args.overlay_width / float(ov.shape[1])
                new_wh = (args.overlay_width, max(1, int(round(ov.shape[0] * scale))))
                ov = cv2.resize(ov, new_wh, interpolation=cv2.INTER_AREA)
            overlay_rgba = ov
            print(f"Overlay loaded: {args.overlay}  → {ov.shape[1]}x{ov.shape[0]}")
    else:
        if args.overlay:
            print(f"WARNING: Overlay path not found: {args.overlay}")

    try:
        stream = model.track(
            source=src,
            device=device,
            imgsz=args.imgsz,
            conf=args.conf,
            stream=True,
            show=False,
            save=False,
            tracker=TRACKER_CFG,
            persist=True,
            verbose=False,
        )

        for result in stream:
            frame = result.plot()
            h, w = frame.shape[:2]

            # Draw geometry
            if mode == "line":
                if len(line_pts) == 2:
                    cv2.line(frame, line_pts[0], line_pts[1], (0, 255, 255), 2)
            else:
                if len(zone_pts) >= 1:
                    for i in range(len(zone_pts)-1):
                        cv2.line(frame, zone_pts[i], zone_pts[i+1], (0, 255, 255), 2)
                    if len(zone_pts) >= 3:
                        cv2.polylines(frame, [np.array(zone_pts, dtype=np.int32)], True, (0, 255, 255), 2)

            # Tracking boxes/ids
            boxes = getattr(result, "boxes", None)
            if boxes is not None and len(boxes) > 0 and boxes.id is not None:
                ids = boxes.id.int().tolist()
                cls_ids = boxes.cls.int().tolist() if boxes.cls is not None else [None]*len(ids)
                xyxy = boxes.xyxy.tolist() if boxes.xyxy is not None else []
                names_map = getattr(result, "names", {})

                zone_mask = None
                if mode == "zone" and len(zone_pts) >= 3:
                    zone_mask = poly_to_mask(zone_pts, w, h)

                for det_id, cls_id, bb in zip(ids, cls_ids, xyxy):
                    cx = int((bb[0] + bb[2]) / 2)
                    cy = int((bb[1] + bb[3]) / 2)
                    cls_name = names_map.get(int(cls_id), str(cls_id))

                    if mode == "line" and len(line_pts) == 2:
                        a, b = line_pts
                        cur_side = side_of_line((cx, cy), a, b)
                        prev_side = last_side.get(det_id, 0)
                        last_side[det_id] = cur_side
                        if prev_side != 0 and cur_side != 0 and np.sign(prev_side) != np.sign(cur_side):
                            if det_id not in counted_tracks:
                                counted_tracks.add(det_id)
                                totals[cls_name] += 1
                                cv2.circle(frame, (cx, cy), 10, (0, 255, 0), 2)
                                print(f"Track {det_id} crossed line -> {cls_name}: {totals[cls_name]}")

                    elif mode == "zone" and zone_mask is not None:
                        inside = bool(zone_mask[cy, cx]) if 0 <= cy < h and 0 <= cx < w else False
                        prev_inside = in_zone.get(det_id, False)
                        in_zone[det_id] = inside
                        if (not prev_inside) and inside:
                            if det_id not in counted_tracks:
                                counted_tracks.add(det_id)
                                totals[cls_name] += 1
                                cv2.circle(frame, (cx, cy), 10, (0, 255, 0), 2)
                                print(f"Track {det_id} ENTERED zone -> {cls_name}: {totals[cls_name]}")

            # HUD
            y = 26
            cv2.putText(frame, f"Mode: {mode.upper()}  (m to toggle)", (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2); y += 24
            if mode == "line":
                if len(line_pts) == 2:
                    cv2.putText(frame, "Line: click twice to set; right-click to clear",
                                (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2); y += 24
                else:
                    cv2.putText(frame, "Set LINE: left-click two points (A then B).",
                                (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2); y += 24
            else:
                cv2.putText(frame, "ZONE: left-click add, right-click close, 'z' undo, 'x' clear",
                            (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2); y += 24

            cv2.putText(frame, f"Prompts: {', '.join(prompts) if prompts else '(open-vocab)'}  (p to edit)",
                        (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180,255,180), 2); y += 24

            if len(totals):
                for k in sorted(totals.keys()):
                    cv2.putText(frame, f"{k}: {totals[k]}", (10, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,220,50), 2); y += 24
            else:
                cv2.putText(frame, "(no counts yet)", (10, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180,180,180), 2); y += 24

            # FPS
            now = time.time()
            dt = now - t_last
            t_last = now
            if dt > 0:
                fps = 1.0/dt
                cv2.putText(frame, f"{fps:.1f} FPS", (w-140, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,255,0), 2)

            # ----- Draw overlay at bottom-left -----
            if overlay_rgba is not None:
                oh, ow = overlay_rgba.shape[:2]
                x = args.overlay_margin
                y = h - oh - args.overlay_margin  # bottom-left
                paste_overlay_rgba(frame, overlay_rgba, x, y)

            cv2.imshow(window, frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                geometry_str = (f"line:{line_pts}" if mode == "line" else f"zone:{zone_pts}")
                safe_save_spreadsheets(args.save_prefix, totals, mode, geometry_str, prompts, device, extra_notes="auto-exit")
                break
            elif key == ord('s'):
                geometry_str = (f"line:{line_pts}" if mode == "line" else f"zone:{zone_pts}")
                safe_save_spreadsheets(args.save_prefix, totals, mode, geometry_str, prompts, device, extra_notes="manual-save")
            elif key == ord('m'):
                mode = "zone" if mode == "line" else "line"
                print("Mode:", mode)
            elif key == ord('r'):
                reset_counts()
            elif key == ord('c'):
                show_centers = not show_centers
                print("Show centers:", show_centers)
            elif key == ord('p'):
                prompt_edit = not prompt_edit
                if prompt_edit:
                    prompt_buffer = ", ".join(prompts)
                else:
                    prompt_buffer = prompt_buffer.strip()
                    apply_prompt_buffer()
            elif prompt_edit:
                if key in (8, 127):
                    prompt_buffer = prompt_buffer[:-1]
                elif key in (13, 10):
                    prompt_edit = False
                    apply_prompt_buffer()
                elif key != 255:
                    if 32 <= key <= 126:
                        prompt_buffer += chr(key)

            if mode == "zone":
                if key == ord('z') and len(zone_pts) > 0:
                    zone_pts.pop()
                elif key == ord('x'):
                    zone_pts.clear()

            if key in [ord(str(d)) for d in range(6)]:
                new_idx = int(chr(key))
                print(f"Switching camera to index {new_idx} ...")
                src = new_idx
                stream = model.track(
                    source=src, device=device, imgsz=args.imgsz, conf=args.conf,
                    stream=True, show=False, save=False, tracker=TRACKER_CFG,
                    persist=True, verbose=False,
                )

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        geometry_str = (f"line:{line_pts}" if mode == "line" else f"zone:{zone_pts}")
        safe_save_spreadsheets(args.save_prefix, totals, mode, geometry_str, prompts, device, extra_notes="keyboard-interrupt")
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

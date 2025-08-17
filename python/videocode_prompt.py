#!/usr/bin/env python
"""
YOLOEtest.py
- YOLOE + CLIP text prompts OR Prompt-Free (PF) mode
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
from pathlib import Path
import hashlib

import numpy as np
import cv2
import torch


# ------------------------- Args ----------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="YOLOE counter with on-screen editing + spreadsheets + overlay.")
    p.add_argument("--model", default="yoloe-11l-seg.pt", help="YOLOE checkpoint (PF or text-promptable)")
    p.add_argument("--source", default="0", help="Webcam index (0/1/2), file path, RTSP/HTTP URL, etc.")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    p.add_argument("--prompts", default= "iphone, fork, remote control", help="Initial prompts (comma-separated). Leave empty for PF.")
    p.add_argument("--save_prefix", default="wasteye_counts", help="Base filename for spreadsheets")
    p.add_argument("--tracker", default="bytetrack.yaml", help="Tracker YAML path (auto-resolve if not found)")
    p.add_argument("--debounce_ms", type=int, default=250, help="Debounce time for line crossing (ms)")
    p.add_argument("--save_video", default="wasteye_counts.mp4", help="Optional path to save annotated video (e.g., out.mp4)")
    p.add_argument("--prompt_mode", choices=["auto", "pf", "text"], default="auto",
                   help="Force prompt-free (pf) or text-prompt mode. 'auto' infers from checkpoint name.")
    p.add_argument("--max_det", type=int, default=300, help="Maximum detections per image (PF may need 1000).")

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
        if torch.cuda.is_available():
            return "cuda"
        return "mps" if torch.backends.mps.is_available() else "cpu"
    return arg_device


def center_of_box(xyxy):
    x1, y1, x2, y2 = xyxy
    return int((x1 + x2) / 2), int((y1 + y2) / 2)


def side_of_line(p, a, b):
    (x, y) = p
    (x1, y1) = a
    (x2, y2) = b
    return np.sign((x2 - x1) * (y - y1) - (y2 - y1) * (x - x1))


def poly_to_mask(poly_pts, w, h):
    mask = np.zeros((h, w), dtype=np.uint8)
    if poly_pts is not None and len(poly_pts) >= 3:
        cv2.fillPoly(mask, [np.array(poly_pts, dtype=np.int32)], 1)
    return mask


def timestamp_str():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def file_md5(p):
    try:
        with open(p, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()[:8]
    except Exception:
        return "NA"


def safe_save_spreadsheets(save_prefix, totals_dict, mode, geometry_str, prompts_list, device, extra_notes=""):
    if not totals_dict:
        print("No counts yet — skipping save.")
        return  # <-- exit early, no files written

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
            "source": str(SAFE_ARGS.source) if 'SAFE_ARGS' in globals() else "",
            "model_name": os.path.basename(SAFE_ARGS.model) if 'SAFE_ARGS' in globals() else "",
            "model_md5": file_md5(SAFE_ARGS.model) if 'SAFE_ARGS' in globals() else "NA",
        })

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
        import pandas as pd
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
    ov = overlay_rgba[oy0:oy1, ox0:ox1]

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
    # Expose to saver for metadata
    global SAFE_ARGS
    SAFE_ARGS = args

    src = int(args.source) if args.source.isdigit() else args.source

    device = pick_device(args.device)
    print("=" * 70)
    print("Python:", sys.executable)
    print(f"Device: {device} (CUDA avail: {torch.cuda.is_available()}, MPS avail: {torch.backends.mps.is_available()})")
    print("Model :", args.model)

    # --- Import YOLOE robustly ---
    YOLOE = None
    import_error = None
    try:
        from ultralytics import YOLOE as _YOLOE
        YOLOE = _YOLOE
    except Exception as e:
        import_error = e
        try:
            from yoloe import YOLOE as _YOLOE  # some setups expose it directly
            YOLOE = _YOLOE
        except Exception as e2:
            import_error = (import_error, e2)
    if YOLOE is None:
        print("Failed to import YOLOE. Ensure the THU-MIG 'yoloe' repo (and weights) are installed.")
        print("Import errors:", import_error)
        sys.exit(1)

    # --- Decide prompt mode ---
    model_basename = os.path.basename(args.model).lower()
    inferred_pf = ("-pf" in model_basename) or ("_pf" in model_basename)
    if args.prompt_mode == "pf":
        use_pf = True
    elif args.prompt_mode == "text":
        use_pf = False
    else:
        use_pf = inferred_pf
    if use_pf and args.conf > 0.05:
        # PF typically benefits from a lower conf; honor user if they set lower than this.
        args.conf = 0.05

    # --- Resolve tracker config ---
    TRACKER_CFG = args.tracker
    if not Path(TRACKER_CFG).exists():
        try:
            from ultralytics.utils import ROOT
            cand = Path(ROOT) / "cfg" / "trackers" / "bytetrack.yaml"
            if cand.exists():
                TRACKER_CFG = str(cand)
        except Exception:
            pass

    # Prompts setup (only for text mode)
    prompts = [s.strip() for s in args.prompts.split(",") if s.strip()]
    clip_ok = True
    if prompts and not use_pf:
        try:
            import clip  # noqa: F401
        except Exception:
            clip_ok = False
            print("WARNING: CLIP not importable. Prompts will NOT filter classes.")
            print("         pip install git+https://github.com/ultralytics/CLIP.git")

    # Warn if a local file path is given but missing (the loader may still resolve a hub name)
    if os.path.sep in args.model and not os.path.exists(args.model):
        print(f"WARNING: Model path not found: {args.model}")

    model = YOLOE(args.model)
    if use_pf:
        print("Running in PROMPT-FREE mode (internal vocabulary).")
        # In PF mode, ignore any prompts string entirely
        prompts = []
        # PF can benefit from higher max_det; try to set on head if available
        try:
            if hasattr(model, "model") and hasattr(model.model[-1], "max_det"):
                model.model[-1].max_det = max(args.max_det, 300)
        except Exception:
            pass
    elif prompts and clip_ok:
        try:
            model.set_classes(prompts, model.get_text_pe(prompts))
            print("Text prompts set:", prompts)
        except Exception as e:
            print("WARNING: Failed to set prompts; running open-vocab. Err:", e)

    totals = defaultdict(int)
    counted = set()         # (track_id, cls_name)
    last_side = {}          # det_id -> {-1,0,+1}
    last_change_t = {}      # det_id -> timestamp
    in_zone = {}
    show_centers = False

    mode = "line"
    line_pts = []
    zone_pts = []
    drawing_zone = False
    zone_mask = None
    zone_pts_version = None

    prompt_edit = False
    prompt_buffer = ""

    window = "YOLOE Counter (GUI)"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    def on_mouse(event, x, y, flags, param):
        nonlocal line_pts, zone_pts, drawing_zone, zone_pts_version, zone_mask
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
        # Invalidate cached zone mask whenever geometry changes
        zone_pts_version = None
        zone_mask = None

    cv2.setMouseCallback(window, on_mouse)

    print("=" * 70)
    print("Hotkeys: q=quit | s=save | m=mode | r=reset | c=centers | p=prompt edit | z=undo (zone) | x=clear (zone)")
    print("Mouse (line):  left-click twice to set A->B, right-click to clear")
    print("Mouse (zone):  left-click to add vertex, right-click to close; 'z' undo, 'x' clear")
    print("=" * 70)

    t_last = time.time()

    def reset_counts():
        totals.clear()
        counted.clear()
        last_side.clear()
        last_change_t.clear()
        in_zone.clear()
        print("Counts reset.")

    def apply_prompt_buffer():
        nonlocal prompts, prompt_buffer
        new_prompts = [s.strip() for s in prompt_buffer.split(",") if s.strip()]
        if len(new_prompts) == 0:
            print("Prompts cleared; running open-vocab.")
            prompts = []
            return
        if use_pf:
            print("PF mode active: prompts ignored.")
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

    # --- Optional video writer ---
    writer = None
    writer_size = None
    writer_fps = 30.0

    try:
        stream = model.track(
            source=src,
            device=device,
            imgsz=args.imgsz,
            conf=args.conf,
            max_det=args.max_det,
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
            if args.save_video:
                if writer is None or writer_size != (w, h):
                    if writer is not None:
                        writer.release()
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(args.save_video, fourcc, writer_fps, (w, h))
                    writer_size = (w, h)

            # Draw geometry
            if mode == "line":
                if len(line_pts) == 2:
                    cv2.line(frame, line_pts[0], line_pts[1], (0, 255, 255), 2)
            else:
                if len(zone_pts) >= 1:
                    for i in range(len(zone_pts) - 1):
                        cv2.line(frame, zone_pts[i], zone_pts[i + 1], (0, 255, 255), 2)
                    if len(zone_pts) >= 3:
                        cv2.polylines(frame, [np.array(zone_pts, dtype=np.int32)], True, (0, 255, 255), 2)

            # Tracking boxes/ids
            boxes = getattr(result, "boxes", None)
            if boxes is not None and len(boxes) > 0 and getattr(boxes, "id", None) is not None:
                ids = boxes.id.int().tolist()
                xyxy = boxes.xyxy.tolist() if boxes.xyxy is not None else []
                if len(ids) != len(xyxy) or len(ids) == 0:
                    pass  # skip weird frames with mismatched arrays
                else:
                    cls_ids = boxes.cls.int().tolist() if boxes.cls is not None else [None] * len(ids)
                    names_map = getattr(result, "names", {})

                    # Cached zone mask
                    if mode == "zone" and len(zone_pts) >= 3:
                        cur_version = hash(tuple(zone_pts))
                        if zone_pts_version != cur_version or zone_mask is None:
                            zone_mask = poly_to_mask(zone_pts, w, h)
                            zone_pts_version = cur_version

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
                                nowt = time.time()
                                if nowt - last_change_t.get(det_id, 0) > (args.debounce_ms / 1000.0):
                                    key = (det_id, cls_name)
                                    if key not in counted:
                                        counted.add(key)
                                        totals[cls_name] += 1
                                        cv2.circle(frame, (cx, cy), 10, (0, 255, 0), 2)
                                        print(f"Track {det_id} crossed line -> {cls_name}: {totals[cls_name]}")
                                last_change_t[det_id] = nowt

                        elif mode == "zone" and zone_mask is not None:
                            inside = bool(zone_mask[cy, cx]) if 0 <= cy < h and 0 <= cx < w else False
                            prev_inside = in_zone.get(det_id, False)
                            in_zone[det_id] = inside
                            if (not prev_inside) and inside:
                                key = (det_id, cls_name)
                                if key not in counted:
                                    counted.add(key)
                                    totals[cls_name] += 1
                                    cv2.circle(frame, (cx, cy), 10, (0, 255, 0), 2)
                                    print(f"Track {det_id} ENTERED zone -> {cls_name}: {totals[cls_name]}")

            # HUD
            y = 26
            cv2.putText(frame, f"Mode: {mode.upper()}  (m to toggle)", (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2); y += 24
            if mode == "line":
                if len(line_pts) == 2:
                    cv2.putText(frame, "Line: click twice to set; right-click to clear",
                                (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2); y += 24
                else:
                    cv2.putText(frame, "Set LINE: left-click two points (A then B).",
                                (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2); y += 24
            else:
                cv2.putText(frame, "ZONE: left-click add, right-click close, 'z' undo, 'x' clear",
                            (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2); y += 24

            cv2.putText(frame, f"Prompts: {', '.join(prompts) if prompts else '(open-vocab)'}  (p to edit)",
                        (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 255, 180), 2); y += 24
            # Status line for PF/text mode
            mode_str = "PF" if use_pf else ("TEXT" if prompts else "OPEN")
            cv2.putText(frame, f"YOLOE mode: {mode_str}",
                        (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 120), 2); y += 24

            if prompt_edit:
                if use_pf:
                    cv2.putText(frame, "(PF mode ignores prompts)", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (90, 90, 255), 2); y += 24
                cv2.rectangle(frame, (8, y), (w - 8, y + 36), (0, 0, 0), -1)
                cv2.putText(frame, f"Edit prompts: {prompt_buffer}_",
                            (14, y + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2); y += 44

            if len(totals):
                for k in sorted(totals.keys()):
                    cv2.putText(frame, f"{k}: {totals[k]}", (10, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 220, 50), 2); y += 24
            else:
                cv2.putText(frame, "(no counts yet)", (10, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 2); y += 24

            # FPS
            now = time.time()
            dt = now - t_last
            t_last = now
            if dt > 0:
                fps = 1.0 / dt
                cv2.putText(frame, f"{fps:.1f} FPS", (w - 140, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 0), 2)
                writer_fps = 0.9 * writer_fps + 0.1 * fps if writer_fps else fps

            # ----- Draw overlay at bottom-left -----
            if overlay_rgba is not None:
                oh, ow = overlay_rgba.shape[:2]
                x = args.overlay_margin
                y0 = h - oh - args.overlay_margin  # bottom-left
                paste_overlay_rgba(frame, overlay_rgba, x, y0)

            # Optional center dots
            if show_centers and boxes is not None and getattr(boxes, "xyxy", None) is not None and getattr(boxes, "id", None) is not None:
                for bb in boxes.xyxy.tolist():
                    cx = int((bb[0] + bb[2]) / 2)
                    cy = int((bb[1] + bb[3]) / 2)
                    cv2.circle(frame, (cx, cy), 3, (0, 255, 255), -1)

            cv2.imshow(window, frame)
            if writer is not None:
                writer.write(frame)
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
                    if use_pf:
                        print("PF mode active: prompts ignored.")
                    else:
                        apply_prompt_buffer()
            elif prompt_edit:
                if key in (8, 127):
                    prompt_buffer = prompt_buffer[:-1]
                elif key in (13, 10):
                    prompt_edit = False
                    if use_pf:
                        print("PF mode active: prompts ignored.")
                    else:
                        apply_prompt_buffer()
                elif key != 255:
                    if 32 <= key <= 126:
                        prompt_buffer += chr(key)

            if mode == "zone":
                if key == ord('z') and len(zone_pts) > 0:
                    zone_pts.pop()
                    zone_pts_version = None; zone_mask = None
                elif key == ord('x'):
                    zone_pts.clear()
                    zone_pts_version = None; zone_mask = None

            if key in [ord(str(d)) for d in range(6)]:
                new_idx = int(chr(key))
                print(f"Switching camera to index {new_idx} ...")
                src = new_idx
                stream = model.track(
                    source=src, device=device, imgsz=args.imgsz, conf=args.conf, max_det=args.max_det,
                    stream=True, show=False, save=False, tracker=TRACKER_CFG,
                    persist=True, verbose=False,
                )

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        geometry_str = (f"line:{line_pts}" if mode == "line" else f"zone:{zone_pts}")
        safe_save_spreadsheets(args.save_prefix, totals, mode, geometry_str, prompts, device, extra_notes="keyboard-interrupt")
    finally:
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

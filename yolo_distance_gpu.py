import math
import json
import os
from typing import Dict

import cv2
import numpy as np
import torch
from ultralytics import YOLO


def load_class_sizes(path: str) -> Dict[str, Dict[str, float]]:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def compute_distance_mm(apparent_pixels: float, real_mm: float, fov_rad: float, frame_width: int) -> float:
    if apparent_pixels <= 0:
        return 0.0
    object_angle = fov_rad * (apparent_pixels / frame_width)
    if object_angle <= 0:
        return 0.0
    distance_mm = real_mm / math.tan(object_angle / 2.0)
    return distance_mm


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    cfg_path = os.path.join(base_dir, 'config_yolo.json')
    with open(cfg_path, 'r', encoding='utf-8') as f:
        cfg = json.load(f)

    class_sizes_path = os.path.join(base_dir, cfg['class_sizes_file'])
    class_sizes = load_class_sizes(class_sizes_path)
    label_overrides = cfg.get('label_overrides_mm', {})

    fov_degree = cfg.get('fov_degree', 50.0) * cfg.get('fov_correction_factor', 1.0)
    fov_rad = fov_degree / 180.0 * math.pi

    model_path = cfg.get('model_path', cfg.get('model', 'yolov8x.pt'))
    model = YOLO(model_path)

    # device selection
    device = cfg.get('device', None)
    if device is None:
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    source = cfg.get('source', 'camera')
    if source == 'camera':
        cap = cv2.VideoCapture(int(cfg.get('camera_number', 0)))
    else:
        video_path = source
        if not os.path.isabs(video_path):
            video_path = os.path.join(base_dir, video_path)
        cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print('Error: cannot open source:', source)
        return 1

    cv2.namedWindow('YOLO Distance (GPU)')

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break

        h, w = frame.shape[:2]

        results = model.predict(source=frame, verbose=False, device=device)[0]

        for b in results.boxes:
            cls_idx = int(b.cls.item())
            conf = float(b.conf.item())
            x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
            bw = max(1, x2 - x1)
            bh = max(1, y2 - y1)

            label = results.names.get(cls_idx, str(cls_idx))
            size_info = label_overrides.get(label) or class_sizes.get(label)
            if size_info is None:
                continue

            use_width = bool(cfg.get('use_width', True))
            real_mm = float(size_info['width_mm' if use_width else 'height_mm'])
            apparent_pixels = float(bw if use_width else bh)

            distance_mm = compute_distance_mm(apparent_pixels, real_mm, fov_rad, w)
            distance_m = distance_mm / 1000.0

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            text = f"{label} {conf:.2f}  {distance_m:.2f}m"
            cv2.putText(frame, text, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.putText(frame, f'device={device}', (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 2)
        cv2.imshow('YOLO Distance (GPU)', frame)
        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())



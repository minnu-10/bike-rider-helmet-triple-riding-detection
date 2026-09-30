import os
os.environ["OPENCV_LOG_LEVEL"] = "ERROR"
import cv2

# ONLY FIX FOR DISPLAY ERROR - save original imshow before ultralytics patches it
_original_imshow = cv2.imshow

import numpy as np
from ultralytics import YOLO
import datetime
import sys
import time
import threading
from collections import deque

# Restore original imshow after ultralytics import
cv2.imshow = _original_imshow

# ================= CONFIG =================
CAMERA_URL = "http://10.111.21.147:8080/"
COOLDOWN_DURATION = 5       # seconds cooldown after violation
VIOLATION_CONFIRM_FRAMES = 3  # violation must appear in N of last 5 frames

# ================= OUTPUT FOLDER =================
current_dir = os.path.dirname(os.path.abspath(__file__))
fines_dir = os.path.join(current_dir, "fines")
os.makedirs(fines_dir, exist_ok=True)

print(f"[INFO] Output folder: {fines_dir}")

# ================= MODELS =================
print("[INFO] Loading models...")

_dir = os.path.dirname(os.path.abspath(__file__))

person_model = YOLO(os.path.join(_dir, "yolov8s.pt"))

helmet_net = cv2.dnn.readNet(
    os.path.join(_dir, "yolov3-helmet.weights"),
    os.path.join(_dir, "yolov3-helmet.cfg")
)
try:
    helmet_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    helmet_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    dummy = cv2.dnn.blobFromImage(
        __import__("numpy").zeros((10,10,3), dtype="uint8"),
        1/255.0, (32, 32)
    )
    helmet_net.setInput(dummy)
    helmet_net.forward(helmet_net.getUnconnectedOutLayersNames())
    print("[INFO] Helmet model: Using GPU (CUDA)")
except Exception:
    helmet_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    helmet_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    print("[INFO] Helmet model: CUDA unavailable, using CPU")

with open(os.path.join(_dir, "helmet.names")) as f:
    helmet_classes = [x.strip().lower() for x in f.readlines()]

print("[INFO] Models loaded successfully!")

# ================= PERFORMANCE MONITORING =================
frame_times = []
adaptive_skip = 1

def update_frame_skip():
    global adaptive_skip, frame_times
    if len(frame_times) > 10:
        avg_time = sum(frame_times[-10:]) / 10
        if avg_time > 0.5:
            adaptive_skip = 5
        elif avg_time > 0.3:
            adaptive_skip = 3
        else:
            adaptive_skip = 1

# ================= BACKGROUND THREAD =================
latest_frame = None
_frame_lock = threading.Lock()
thread_running = True

def frame_reader(url):
    global latest_frame, thread_running
    base_url = url.rstrip("/")
    cap = None
    endpoints = [
        f"{base_url}/video",
        f"{base_url}/videofeed",
        f"{base_url}/video?640x480",
    ]
    for endpoint in endpoints:
        for backend in [cv2.CAP_FFMPEG, cv2.CAP_ANY, None]:
            try:
                backend_name = {cv2.CAP_FFMPEG: "FFMPEG", cv2.CAP_ANY: "CAP_ANY"}.get(backend, "Default")
                print(f"  [{backend_name}] {endpoint}...", end=" ")
                temp_cap = cv2.VideoCapture(endpoint, backend) if backend else cv2.VideoCapture(endpoint)
                if temp_cap.isOpened():
                    ret, test_frame = temp_cap.read()
                    if ret and test_frame is not None:
                        cap = temp_cap
                        print("OK")
                        break
                    temp_cap.release()
                print("FAIL")
            except:
                print("FAIL")
                continue
        if cap is not None:
            break

    if cap is None:
        print("\nAll connection attempts failed")
        thread_running = False
        return

    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    while thread_running:
        ret, frame = cap.read()
        if ret:
            with _frame_lock:
                latest_frame = frame
        else:
            time.sleep(0.1)
    cap.release()

print("[INFO] Connecting to camera...")
reader_thread = threading.Thread(target=frame_reader, args=(CAMERA_URL,), daemon=True)
reader_thread.start()
time.sleep(3)

if latest_frame is None:
    print("\nCamera not reachable!")
    print("  1. Open IP Webcam -> Start Server")
    print("  2. Check IP: 192.168.31.52:8080")
    print("  3. Same WiFi for phone & PC")
    input("\nPress Enter to exit...")
    sys.exit(1)

print("Camera connected!\n")

# ================= DETECTION HELPERS =================

def compute_iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 < ix1 or iy2 < iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def overlap(person_bbox, vehicle_bbox):
    px1, py1, px2, py2 = person_bbox
    vx1, vy1, vx2, vy2 = vehicle_bbox

    pc_x = (px1 + px2) / 2
    pc_y = (py1 + py2) / 2

    expand = 20
    centroid_inside = (vx1 - expand <= pc_x <= vx2 + expand and
                       vy1 - expand <= pc_y <= vy2 + expand)

    iou = compute_iou(person_bbox, vehicle_bbox)
    iou_overlap = iou > 0.05

    ix1, iy1 = max(px1, vx1), max(py1, vy1)
    ix2, iy2 = min(px2, vx2), min(py2, vy2)
    person_area = (px2 - px1) * (py2 - py1)
    if ix2 > ix1 and iy2 > iy1 and person_area > 0:
        inter = (ix2 - ix1) * (iy2 - iy1)
        area_ratio = inter / person_area
    else:
        area_ratio = 0.0

    area_overlap = area_ratio > 0.20
    bottom_in_vehicle = py2 >= vy1 and py1 <= vy2 + 30

    return (centroid_inside or iou_overlap or area_overlap) and bottom_in_vehicle


def horizontal_distance(person_bbox, vehicle_bbox):
    px1, _, px2, _ = person_bbox
    vx1, _, vx2, _ = vehicle_bbox
    if px2 < vx1:
        return vx1 - px2
    elif px1 > vx2:
        return px1 - vx2
    return 0

def vertical_distance(person_bbox, vehicle_bbox):
    _, py1, _, py2 = person_bbox
    _, vy1, _, vy2 = vehicle_bbox
    if py2 < vy1:
        return vy1 - py2
    elif py1 > vy2:
        return py1 - vy2
    return 0

def is_pedestrian(person_bbox, vehicle_bboxes, frame_height=None, frame_width=None):
    if not vehicle_bboxes:
        return True

    for v in vehicle_bboxes:
        if overlap(person_bbox, v):
            return False

        h_dist = horizontal_distance(person_bbox, v)
        v_dist = vertical_distance(person_bbox, v)
        vw = v[2] - v[0]
        vh = v[3] - v[1]

        if h_dist < vw * 0.15 and v_dist < vh * 0.20:
            return False

    return True

def is_child(bbox, frame_height, frame_width, co_riders=None):
    """
    Detects if a rider is a child.

    FIX: size_ratio veto changed from 0.90 → 0.85.
    Adults in a group stay within 85-100% of each other's height.
    The old 0.90 threshold mis-classified the middle rider in triple-riding
    (ratio ~0.89) as a child. Real children are typically <75% of adult height,
    so 0.85 safely separates the two cases.
    """
    x1, y1, x2, y2 = bbox
    person_height = y2 - y1
    person_width  = x2 - x1
    person_area   = person_height * person_width
    height_ratio  = person_height / frame_height
    width_ratio   = person_width  / frame_width

    score = 0

    # --- Absolute size (works at normal / far distances) ---
    if height_ratio < 0.18:   score += 3
    elif height_ratio < 0.25: score += 2
    elif height_ratio < 0.32: score += 1

    if person_height < 110:   score += 3
    elif person_height < 140: score += 2
    elif person_height < 170: score += 1

    if person_area < 7000:    score += 2
    elif person_area < 12000: score += 1

    if person_height > 0:
        aspect = person_width / person_height
        if 0.55 <= aspect <= 0.90: score += 2
        elif aspect > 0.90:        score += 1

    if width_ratio < 0.06: score += 1

    # --- Relative size vs co-riders (works at close-up distances) ---
    if co_riders and len(co_riders) > 0:
        other_heights = [cr[3] - cr[1] for cr in co_riders
                         if (cr[0], cr[1], cr[2], cr[3]) != (x1, y1, x2, y2)]

        if other_heights:
            avg_other_height = sum(other_heights) / len(other_heights)
            size_ratio = person_height / avg_other_height if avg_other_height > 0 else 1.0

            # Relative veto: >= 85% of co-rider average height = adult.
            # Changed from 0.90: middle rider in triple-riding had ratio ~0.89,
            # causing false CHILD classification. 0.85 fixes this.
            if size_ratio >= 0.85:
                return False

            # Scoring (added <0.55 tier for very-close-up small children)
            if size_ratio < 0.55:   score += 5
            elif size_ratio < 0.60: score += 4
            elif size_ratio < 0.75: score += 3
            elif size_ratio < 0.82: score += 2
            elif size_ratio < 0.85: score += 1

    # Head-body proportion bonus (unchanged)
    if person_height > 0:
        head_body_ratio = (person_height * 0.22) / person_height
        if head_body_ratio >= 0.22 and person_height < 160:
            score += 1

    return score >= 3


def classify_riders_by_relative_size(rider_bboxes, frame_height, frame_width):
    if not rider_bboxes:
        return [], []
    adults = []
    children = []
    for bbox in rider_bboxes:
        if is_child(bbox, frame_height, frame_width, co_riders=rider_bboxes):
            children.append(bbox)
        else:
            adults.append(bbox)
    return adults, children


def detect_helmets_yolov3(image, net, class_names, conf_threshold=0.4, nms_threshold=0.4):
    if image is None or image.size == 0 or image.shape[0] < 5 or image.shape[1] < 5:
        return []
    h, w = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    out_names = net.getUnconnectedOutLayersNames()
    outputs = net.forward(out_names)
    boxes = []; confidences = []; class_ids = []
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = int(np.argmax(scores))
            confidence = float(scores[class_id])
            if confidence > conf_threshold:
                center_x = int(detection[0] * w); center_y = int(detection[1] * h)
                bw = int(detection[2] * w); bh = int(detection[3] * h)
                x = int(center_x - bw / 2); y = int(center_y - bh / 2)
                boxes.append([x, y, bw, bh]); confidences.append(confidence); class_ids.append(class_id)
    idxs = []
    if len(boxes) > 0:
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    detections = []
    if len(idxs) > 0:
        for i in idxs.flatten():
            cls_name = class_names[class_ids[i]] if class_ids[i] < len(class_names) else str(class_ids[i])
            detections.append({"box": boxes[i], "conf": confidences[i], "class_id": class_ids[i], "label": cls_name})
    return detections


def detect_helmets_on_frame(frame, conf_threshold=0.35, nms_threshold=0.4):
    if frame is None or frame.size == 0:
        return []
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    helmet_net.setInput(blob)
    out_names = helmet_net.getUnconnectedOutLayersNames()
    outputs = helmet_net.forward(out_names)
    boxes, confidences = [], []
    for output in outputs:
        for detection in output:
            scores     = detection[5:]
            class_id   = int(np.argmax(scores))
            confidence = float(scores[class_id])
            if confidence > conf_threshold:
                if class_id < len(helmet_classes) and "helmet" in helmet_classes[class_id]:
                    cx = int(detection[0] * w); cy = int(detection[1] * h)
                    bw = int(detection[2] * w); bh = int(detection[3] * h)
                    boxes.append([cx - bw//2, cy - bh//2, bw, bh])
                    confidences.append(confidence)
    if not boxes:
        return []
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    if len(idxs) == 0:
        return []
    helmets = []
    for i in idxs.flatten():
        x, y, bw, bh = boxes[i]
        helmets.append((x + bw // 2, y + bh // 2))
    return helmets


def rider_has_helmet(rider_bbox, helmet_centers):
    x1, y1, x2, y2 = rider_bbox
    head_bottom = y1 + int(0.40 * (y2 - y1))
    for (cx, cy) in helmet_centers:
        if x1 <= cx <= x2 and y1 <= cy <= head_bottom:
            return True
    return False


def merge_vehicle_boxes(vehicles, iou_thresh=0.3):
    if len(vehicles) <= 1:
        return vehicles
    def box_iou(a, b):
        ax1,ay1,ax2,ay2 = a; bx1,by1,bx2,by2 = b
        ix1,iy1 = max(ax1,bx1),max(ay1,by1)
        ix2,iy2 = min(ax2,bx2),min(ay2,by2)
        if ix2<=ix1 or iy2<=iy1: return 0.0
        inter=(ix2-ix1)*(iy2-iy1)
        return inter/((ax2-ax1)*(ay2-ay1)+(bx2-bx1)*(by2-by1)-inter)
    merged = list(vehicles)
    changed = True
    while changed:
        changed = False; result = []; used = [False]*len(merged)
        for i in range(len(merged)):
            if used[i]: continue
            cur = list(merged[i])
            for j in range(i+1, len(merged)):
                if used[j]: continue
                if box_iou(cur, merged[j]) > iou_thresh:
                    cur = [min(cur[0],merged[j][0]),min(cur[1],merged[j][1]),
                           max(cur[2],merged[j][2]),max(cur[3],merged[j][3])]
                    used[j] = True; changed = True
            result.append(tuple(cur)); used[i] = True
        merged = result
    return merged


def apply_nms_to_persons(persons, scores, iou_thresh=0.5):
    if not persons:
        return []
    boxes = [[x1, y1, x2 - x1, y2 - y1] for (x1, y1, x2, y2) in persons]
    indices = cv2.dnn.NMSBoxes(boxes, scores, score_threshold=0.0, nms_threshold=iou_thresh)
    if len(indices) == 0:
        return []
    if isinstance(indices[0], (list, np.ndarray)):
        indices = [i[0] for i in indices]
    return [persons[i] for i in indices]


def count_riders_on_vehicle(vehicle_bbox, person_bboxes, frame_height, frame_width):
    all_riders = [p for p in person_bboxes if overlap(p, vehicle_bbox)]
    if not all_riders:
        return [], []
    adults, children = classify_riders_by_relative_size(all_riders, frame_height, frame_width)
    return adults, children


def generate_fine_with_proof(reason, amount, frame, violation_type):
    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    screenshot_path = os.path.join(fines_dir, f"{violation_type}_proof_{ts}.jpg")
    cv2.imwrite(screenshot_path, frame)
    fine_path = os.path.join(fines_dir, f"fine_{ts}.txt")
    try:
        dt_str = datetime.datetime.now().strftime("%d-%m-%Y %H:%M:%S")
        with open(fine_path, "w", encoding="utf-8") as f:
            f.write("=" * 60 + "\n")
            f.write("         TRAFFIC VIOLATION FINE\n")
            f.write("=" * 60 + "\n")
            f.write(f"Date & Time    : {dt_str}\n")
            f.write(f"Violation Type : {violation_type.upper()}\n")
            f.write(f"Description    : {reason}\n")
            f.write(f"Fine Amount    : INR {amount}\n")
            f.write(f"Evidence Photo : {screenshot_path}\n")
            f.write("=" * 60 + "\n")
        print(f"[FINE ISSUED] {violation_type.upper()} | INR {amount}")
        print(f"   Reason: {reason}")
        print(f"   Files: {os.path.basename(fine_path)} + {os.path.basename(screenshot_path)}\n")
    except Exception as e:
        print(f"[ERROR] Failed to save fine: {e}")


def draw_stats(img, stats, status_text, status_color, border_color):
    H, W = img.shape[:2]
    cv2.rectangle(img, (5, 5), (W - 5, H - 5), border_color, 5)
    cv2.putText(img, status_text, (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, status_color, 3)
    labels = [
        (f"Vehicles      : {stats['vehicles']}", (255, 255, 255)),
        (f"Adult Riders  : {stats['riders']}", (255, 255, 255)),
        (f"Children      : {stats['children']}", (0, 200, 255)),
        (f"Pedestrians   : {stats['pedestrians']}", (200, 200, 0)),
        (f"With Helmet   : {stats['helmet']}", (0, 255, 0)),
        (f"No Helmet     : {stats['no_helmet']}", (0, 0, 255)),
        (f"Triple Riding : {stats['triple']}", (255, 0, 255)),
    ]
    y_pos = H - (len(labels) * 30) - 15
    for text, color in labels:
        cv2.putText(img, text, (10, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        y_pos += 30
    return img


# ================= TEMPORAL SMOOTHING =================
violation_history = deque(maxlen=5)

def is_violation_confirmed(v_type):
    count = sum(1 for frame_violations in violation_history if v_type in frame_violations)
    return count >= VIOLATION_CONFIRM_FRAMES


# ================= MAIN DETECTION LOOP =================
print("=" * 60)
print("  SMART HELMET & TRIPLE RIDING DETECTION SYSTEM v3")
print("=" * 60)
print("Detects: No Helmet | Triple Riding | Combined")
print("Ignores: Children | Pedestrians")
print("Temporal smoothing (3/5 frames): Enabled")
print("Enhanced child detection: Relative size comparison")
print(f"Cooldown: {COOLDOWN_DURATION} seconds (real time) after violation")
print("=" * 60)
print("Press ESC or Q to exit\n")

frame_count = 0
cooldown_end_time = 0

while True:
    with _frame_lock:
        frame = latest_frame
    if frame is None:
        time.sleep(0.1)
        continue

    frame_count += 1

    # ---- Real-time cooldown check ----
    now = time.time()
    if now < cooldown_end_time:
        remaining_sec = int(cooldown_end_time - now) + 1
        display = frame.copy()
        H, W = display.shape[:2]
        cv2.rectangle(display, (5, 5), (W - 5, H - 5), (0, 165, 255), 5)
        cv2.putText(display, f"COOLDOWN: {remaining_sec}s", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 165, 255), 3)
        cv2.putText(display, "Waiting before next scan...", (10, H - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow("Live Detection", display)
        if cv2.waitKey(1) & 0xFF in [27, ord('q'), ord('Q')]:
            break
        continue

    # ---- Adaptive frame skipping ----
    if frame_count % adaptive_skip != 0:
        cv2.imshow("Live Detection", frame)
        if cv2.waitKey(1) & 0xFF in [27, ord('q'), ord('Q')]:
            break
        continue

    # ===== PROCESS FRAME =====
    process_start = time.time()

    scale = 640 / frame.shape[1]
    resized = cv2.resize(frame, None, fx=scale, fy=scale)
    H, W = resized.shape[:2]
    img = resized.copy()

    results = person_model(img, conf=0.40, iou=0.45, verbose=False)[0]

    raw_persons = []
    person_scores = []
    vehicles = []

    for box in results.boxes:
        cls = person_model.names[int(box.cls[0])]
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf_val = float(box.conf[0])

        if cls == "person":
            raw_persons.append((x1, y1, x2, y2))
            person_scores.append(conf_val)
        elif cls in ["motorcycle", "bicycle"]:
            vehicles.append((x1, y1, x2, y2))

    persons = apply_nms_to_persons(raw_persons, person_scores)
    vehicles = merge_vehicle_boxes(vehicles, iou_thresh=0.3)

    if len(vehicles) == 0:
        violation_history.append(set())
        cv2.rectangle(img, (5, 5), (W - 5, H - 5), (0, 255, 0), 3)
        cv2.putText(img, "MONITORING...", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        cv2.putText(img, "Waiting for vehicles...", (10, H - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow("Live Detection", img)
        if cv2.waitKey(1) & 0xFF in [27, ord('q'), ord('Q')]:
            break
        continue

    # ===== SEPARATE PEDESTRIANS =====
    pedestrians = [p for p in persons if is_pedestrian(p, vehicles, frame_height=H, frame_width=W)]
    for p in pedestrians:
        x1, y1, x2, y2 = p
        cv2.rectangle(img, (x1, y1), (x2, y2), (200, 200, 0), 2)
        cv2.putText(img, "PEDESTRIAN", (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 0), 2)

    # ===== ANALYZE VEHICLES =====
    current_frame_violations = set()
    candidate_violations = []
    rider_count = 0
    child_count = 0
    helmet_count = 0
    no_helmet_count = 0
    triple_count = 0

    # Detect all helmets ONCE on full frame — accurate, no crop artifacts
    helmet_centers = detect_helmets_on_frame(img)

    for vehicle in vehicles:
        vx1, vy1, vx2, vy2 = vehicle
        cv2.rectangle(img, (vx1, vy1), (vx2, vy2), (255, 255, 0), 3)
        cv2.putText(img, "VEHICLE", (vx1, vy1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        adults, children = count_riders_on_vehicle(vehicle, persons, H, W)

        is_triple = len(adults) > 2
        if is_triple:
            triple_count += 1

        child_count += len(children)

        for p in children:
            x1, y1, x2, y2 = p
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 200, 255), 4)
            label = "CHILD (EXEMPT)"
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(img, (x1, y1 - lh - 15), (x1 + lw + 10, y1), (0, 200, 255), -1)
            cv2.putText(img, label, (x1 + 5, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        # Sort adults by X position — index 0 = frontmost = DRIVER
        adults_sorted = sorted(adults, key=lambda b: (b[0] + b[2]) / 2)
        driver_no_helmet = False

        for idx, p in enumerate(adults_sorted):
            x1, y1, x2, y2 = p
            rider_count += 1

            has_helmet = rider_has_helmet(p, helmet_centers)

            if has_helmet:
                helmet_count += 1
                color, lbl = (0, 255, 0), "HELMET OK"
            else:
                no_helmet_count += 1
                color, lbl = (0, 0, 255), "NO HELMET"
                if idx == 0:
                    driver_no_helmet = True

            cv2.rectangle(img, (x1, y1), (x2, y2), color, 4)
            cv2.putText(img, lbl, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        if is_triple and driver_no_helmet:
            v_type = "triple_no_helmet"
            candidate_violations.append((
                v_type,
                f"Triple riding ({len(adults)} adult riders) + driver without helmet",
                3000
            ))
            current_frame_violations.add(v_type)
        elif is_triple:
            v_type = "triple_riding"
            candidate_violations.append((
                v_type,
                f"Triple riding ({len(adults)} adult riders on vehicle)",
                2000
            ))
            current_frame_violations.add(v_type)
        elif driver_no_helmet:
            v_type = "no_helmet"
            candidate_violations.append((
                v_type,
                f"Driver riding without helmet",
                1000
            ))
            current_frame_violations.add(v_type)

    violation_history.append(current_frame_violations)

    confirmed_violations = [
        v for v in candidate_violations
        if is_violation_confirmed(v[0])
    ]

    stats = {
        "vehicles": len(vehicles),
        "riders": rider_count,
        "children": child_count,
        "pedestrians": len(pedestrians),
        "helmet": helmet_count,
        "no_helmet": no_helmet_count,
        "triple": triple_count
    }

    if confirmed_violations:
        img = draw_stats(img, stats, "VIOLATION DETECTED!", (0, 0, 255), (0, 0, 255))
        for v_type, v_reason, v_amount in confirmed_violations:
            generate_fine_with_proof(v_reason, v_amount, img, v_type)
        cooldown_end_time = time.time() + COOLDOWN_DURATION
        violation_history.clear()
    else:
        if current_frame_violations:
            img = draw_stats(img, stats, f"CONFIRMING... ({len(violation_history)}/5)", (0, 165, 255), (0, 165, 255))
        else:
            img = draw_stats(img, stats, "ALL CLEAR", (0, 255, 0), (0, 255, 0))

    cv2.imshow("Live Detection", img)

    process_time = time.time() - process_start
    frame_times.append(process_time)
    if len(frame_times) > 30:
        frame_times.pop(0)
    update_frame_skip()

    if cv2.waitKey(1) & 0xFF in [27, ord('q'), ord('Q')]:
        break

# Cleanup
thread_running = False
cv2.destroyAllWindows()
print("\n[INFO] System stopped")
print(f"[INFO] All files saved in: {fines_dir}")